"""
Flask backend: xem chuỗi ảnh DICOM (CT) và vẽ segmentation với NHIỀU mask động
(không giới hạn 3 màu), flood fill (skimage) hoặc khoanh vùng thủ công (lasso).

Kiến trúc dữ liệu (MaskStore):
- meta[mask_id]      -> {"label", "color": (r,g,b), "visible": bool, "status": "buildin"|"custom"}
                                                                              (global, dùng chung mọi slide)
- filters[mask_id]   -> {"include": [mask_id,...], "exclude": [mask_id,...]} (gắn theo mask, dùng khi sửa mask đó)
- pixels[slice_idx][mask_id] -> np.uint8 (H,W) 0/1                           (lazy-init = mask rỗng)
- slide_names[slice_idx] -> str                                             (tên tùy chỉnh cho 1 slide)

Status "buildin" vs "custom":
- Khi khởi tạo MaskStore, tự động tạo sẵn 3 mask rỗng "buildin" (3 màu mặc định) để dùng ngay.
- Mask "custom" là mask người dùng tự tạo thêm (nút ➕).
- Mask "buildin" không thể bị xóa (delete_mask trả về False), luôn được hiển thị ở CUỐI danh sách
  nhưng vẫn có thể đổi tên/màu/ẩn-hiện như mask thường.

Công cụ vẽ: flood fill (tự động lan theo HU), khoanh tay thủ công (polygon), và
brush (vẽ tự do bằng vòng tròn theo đường di chuột, có brush size).

Luồng sửa mask (route /mask_modify):
  existing = MaskStore.get_mask(slice, mask_id)          # lưu lại làm undo
  raw      = render_raw_img(slice)                       # ảnh HU windowed hiện tại
  handled  = mask_handle(raw, op, exclude_arrays, include_arrays)
  new_mask = mask_muxing(existing, handled, mask_mode)   # add/subtract
  MaskStore.set_mask(slice, mask_id, new_mask)
  -> render_final_img(slice) trả về FE

Undo (Ctrl+Z, route /undo_mask): chỉ áp dụng cho /mask_modify, chỉ 1 bước duy nhất,
kiểm tra đúng (slice_index, mask_id) đang thao tác thì mới phục hồi.
"""
import io
import os
import json
import zipfile
import tempfile
import shutil
import base64

import numpy as np
import pydicom
from pydicom.sr.coding import Code
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file

from skimage.segmentation import flood
from skimage.draw import polygon as sk_polygon, line as sk_line
from skimage.morphology import dilation, erosion, disk
from skimage.measure import label as sk_label, regionprops


app = Flask(__name__)

MASK_ALPHA = 0.45
DEFAULT_PALETTE = [
    (255, 0, 0), (255, 210, 0), (0, 220, 90), (0, 170, 255),
    (255, 0, 200), (255, 140, 0), (150, 80, 255), (0, 230, 210),
]


# ----------------------------------------------------------------------------
# MaskStore: object quản lý toàn bộ mask (nhiều mask, nhiều slide)
# ----------------------------------------------------------------------------
class MaskStore:
    def __init__(self, num_slices, h, w):
        self.num_slices = num_slices
        self.h = h
        self.w = w
        self.meta = {}          # mask_id -> {label, color, visible, status}
        self.filters = {}       # mask_id -> {include:[], exclude:[]}
        self.pixels = {}        # slice_idx -> {mask_id: np.uint8 (H,W)}
        self.slide_names = {}   # slice_idx(int) -> str
        self._next_id = 1
        # tạo sẵn 3 mask buildin rỗng (3 màu mặc định) để dùng ngay
        self.create_mask(label = "Cơ xương", color=DEFAULT_PALETTE[0], status="custom")
        self.create_mask(label = "Mỡ dưới da", color=DEFAULT_PALETTE[1], status="custom")
        self.create_mask(label = "Mỡ nội tạng", color=DEFAULT_PALETTE[2], status="custom")

        # Tạo các mask buildin sẵn
        self.create_mask(label = "_cơ_xương", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[29-24,29+24]}, {"opening":2}])
        self.create_mask(label = "_mỡ_dưới_da", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[0-93-23,0-93+23]}, {"opening":4}])
        self.create_mask(label = "_mỡ_nội_tạng", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[0-100,0-50]}, {"opening":1}])
        self.create_mask(label = "_khí", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[-100000000,-200]}, {"opening":2}])

    # ---- vòng đời mask ----
    def create_mask(self, label=None, color=None, status="custom", visible=True, special=None):
        mid = f"m{self._next_id}"
        self._next_id += 1
        if color is None:
            color = DEFAULT_PALETTE[len(self.meta) % len(DEFAULT_PALETTE)]
        if label is None:
            label = f"Mask {mid[1:]}"
        if status not in ("buildin", "custom"):
            status = "custom"
        self.meta[mid] = {
            "label": label, "color": tuple(int(c) for c in color),
            "visible": visible, "status": status, "special": special
        }
        self.filters[mid] = {"include": [], "exclude": []}
        return mid

    def delete_mask(self, mask_id):
        """Xóa mask 'custom'. Mask 'buildin' không thể bị xóa -> trả về False."""
        meta = self.meta.get(mask_id)
        if not meta:
            return False
        if meta.get("status") == "buildin":
            return False
        self.meta.pop(mask_id, None)
        self.filters.pop(mask_id, None)
        for sl in self.pixels.values():
            sl.pop(mask_id, None)
        for f in self.filters.values():
            if mask_id in f["include"]:
                f["include"].remove(mask_id)
            if mask_id in f["exclude"]:
                f["exclude"].remove(mask_id)
        return True

    def list_mask_ids(self):
        """Thứ tự hiển thị: mask 'custom' trước, mask 'buildin' luôn ở cuối."""
        custom = [mid for mid, m in self.meta.items() if m.get("status") != "buildin"]
        buildin = [mid for mid, m in self.meta.items() if m.get("status") == "buildin"]
        return custom + buildin

    def rename_mask(self, mask_id, label):
        if mask_id in self.meta:
            self.meta[mask_id]["label"] = label

    def recolor_mask(self, mask_id, color):
        if mask_id in self.meta:
            self.meta[mask_id]["color"] = tuple(int(c) for c in color)

    def set_visibility(self, mask_id, visible):
        if mask_id in self.meta:
            self.meta[mask_id]["visible"] = bool(visible)

    def exists(self, mask_id):
        return mask_id in self.meta

    # ---- đọc/ghi pixel (an toàn: tự tạo mask rỗng nếu chưa có) ----
    def get_mask(self, slice_idx, mask_id):
        if mask_id not in self.meta:
            raise KeyError(f"Mask '{mask_id}' không tồn tại.")
        sl = self.pixels.setdefault(slice_idx, {})

        if mask_id not in sl:
            meta = self.meta[mask_id]
            if meta.get("status") == "buildin" and meta.get("special") is not None:
                raw_img = STATE["raw_hu"][slice_idx]
                temp_special_mask = np.zeros((self.h, self.w), dtype=np.uint8)
                
                # Duyệt qua từng dict trong list special
                for op_dict in meta["special"]:
                    for op_key, op_value in op_dict.items():
                        if op_key == "threshold":
                            hu_min, hu_max = op_value[0], op_value[1]
                            temp_special_mask = mask_threshold(raw_img, hu_min, hu_max)
                
                        elif op_key in ["opening", "closing"]:
                            temp_special_mask = mask_open_close(temp_special_mask, op_key, op_value)

                sl[mask_id] = temp_special_mask
            else:
                sl[mask_id] = np.zeros((self.h, self.w), dtype=np.uint8)
        return sl[mask_id]

    def set_mask(self, slice_idx, mask_id, array):
        sl = self.pixels.setdefault(slice_idx, {})
        sl[mask_id] = array.astype(np.uint8)

    def clear_slice(self, slice_idx):
        self.pixels[slice_idx] = {}

    def clear_mask_on_slice(self, slice_idx, mask_id):
        if mask_id in self.meta:
            self.pixels.setdefault(slice_idx, {})[mask_id] = np.zeros((self.h, self.w), dtype=np.uint8)

    def get_display_masks(self, slice_idx):
        """Dict mask_id -> {label,color,visible,array} - dùng cho render_final_img."""
        out = {}
        for mid in self.list_mask_ids():
            meta = self.meta[mid]
            out[mid] = {
                "label": meta["label"], "color": meta["color"], "visible": meta["visible"],
                "status": meta.get("status", "custom"),
                "array": self.get_mask(slice_idx, mid),
            }
        return out

    # ---- filter include/exclude gắn theo mask ----
    def set_filters(self, mask_id, include=None, exclude=None):
        f = self.filters.setdefault(mask_id, {"include": [], "exclude": []})
        if include is not None:
            f["include"] = [m for m in include if m in self.meta and m != mask_id]
        if exclude is not None:
            f["exclude"] = [m for m in exclude if m in self.meta and m != mask_id]

    def get_filters(self, mask_id):
        return self.filters.get(mask_id, {"include": [], "exclude": []})

    # ---- tổng hợp thông tin ----
    def list_slices_with_mask(self):
        result = []
        for sl_idx, d in self.pixels.items():
            for mask_id, arr in d.items():
                mask_meta = self.meta.get(mask_id)
                if mask_meta and mask_meta.get("status") == "custom" and arr.any():
                    result.append(int(sl_idx))
                    break
        return sorted(result)

    def to_summary(self):
        return [
            {
                "id": mid,
                "label": self.meta[mid]["label"],
                "color": list(self.meta[mid]["color"]),
                "visible": self.meta[mid]["visible"],
                "status": self.meta[mid].get("status", "custom"),
                "include": self.filters.get(mid, {}).get("include", []),
                "exclude": self.filters.get(mid, {}).get("exclude", []),
            }
            for mid in self.list_mask_ids()
        ]

    # ---- tên slide (đặt tên tùy chỉnh cho 1 slide) ----
    def set_slide_name(self, slice_idx, name):
        name = (name or "").strip()
        slice_idx = int(slice_idx)
        if name:
            self.slide_names[slice_idx] = name
        else:
            self.slide_names.pop(slice_idx, None)

    def delete_slide_name(self, slice_idx):
        self.slide_names.pop(int(slice_idx), None)

    def get_slide_name(self, slice_idx):
        return self.slide_names.get(int(slice_idx), "")

    def list_slice_entries(self):
        """Danh sách slide có mask VÀ/HOẶC có tên, sắp theo index, kèm tên (nếu có)."""
        mask_slices = set(self.list_slices_with_mask())
        name_slices = set(self.slide_names.keys())
        all_idx = sorted(mask_slices | name_slices)
        return [
            {"index": idx, "name": self.slide_names.get(idx, ""), "has_mask": idx in mask_slices}
            for idx in all_idx
        ]


# ----------------------------------------------------------------------------
# Global in-memory state
# ----------------------------------------------------------------------------
STATE = {
    "raw_hu": None,
    "datasets": None,
    "mask_store": None,
    "num_slices": 0,
    "height": 0,
    "width": 0,
    "pixel_spacing": (1.0, 1.0),
    "hu_min": -160.0,
    "hu_max": 240.0,
    "hu_bounds": (-1024.0, 3071.0),
    "case_name": "",
    "undo": {"mask_id": None, "slice_index": None, "mask_array": None},
}


# ----------------------------------------------------------------------------
# Helpers: đọc DICOM
# ----------------------------------------------------------------------------
def _get_first(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) or (hasattr(value, "__len__") and not isinstance(value, (int, float))):
        try:
            return float(value[0])
        except (TypeError, IndexError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dicom_series_from_zip(zip_path):
    tmp_dir = tempfile.mkdtemp(prefix="dcm_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        dcm_paths = []
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if not fn.startswith("."):
                    dcm_paths.append(os.path.join(root, fn))

        datasets = []
        for p in dcm_paths:
            try:
                ds = pydicom.dcmread(p, force=True)
                if not hasattr(ds, "pixel_array"):
                    continue
                datasets.append(ds)
            except Exception:
                continue

        if not datasets:
            raise ValueError("Không tìm thấy file DICOM hợp lệ trong zip.")

        def sort_key(ds):
            if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) == 3:
                return float(ds.ImagePositionPatient[2])
            if hasattr(ds, "InstanceNumber"):
                return float(ds.InstanceNumber)
            return 0.0

        datasets.sort(key=sort_key)

        first = datasets[0]
        wc = _get_first(getattr(first, "WindowCenter", None), 40.0)
        ww = _get_first(getattr(first, "WindowWidth", None), 400.0)

        ps = getattr(first, "PixelSpacing", None)
        if ps and len(ps) == 2:
            pixel_spacing = (float(ps[0]), float(ps[1]))
        else:
            pixel_spacing = (1.0, 1.0)

        hu_slices = []
        h0, w0 = datasets[0].pixel_array.shape
        kept_datasets = []
        for ds in datasets:
            arr = ds.pixel_array
            if arr.shape != (h0, w0):
                continue
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = arr.astype(np.float32) * slope + intercept
            hu_slices.append(hu)
            kept_datasets.append(ds)

        raw_hu = np.stack(hu_slices, axis=0)
        return raw_hu, kept_datasets, pixel_spacing, wc, ww
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------------
# Render chuyên môn hóa: render_raw_img / render_final_img
# ----------------------------------------------------------------------------
def render_raw_img(slice_idx):
    """DICOM HU -> ảnh xám uint8 theo cửa sổ HU hiện tại (hu_min,hu_max)."""
    raw = STATE["raw_hu"][slice_idx]
    lo, hi = STATE["hu_min"], STATE["hu_max"]
    windowed = np.clip(raw, lo, hi)
    windowed = (windowed - lo) / max(hi - lo, 1e-6) * 255.0
    return windowed.astype(np.uint8)


def render_final_img(slice_idx):
    """Ảnh raw + trộn màu các mask đang visible -> base64 PNG."""
    gray = render_raw_img(slice_idx)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)

    mask_store = STATE["mask_store"]
    for mid, info in mask_store.get_display_masks(slice_idx).items():
        if not info["visible"]:
            continue
        m = info["array"]
        if not m.any():
            continue
        color = np.array(info["color"], dtype=np.float64)
        idx_mask = m.astype(bool)
        rgb[idx_mask] = (1 - MASK_ALPHA) * rgb[idx_mask] + MASK_ALPHA * color

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def brush_path_to_region(points, brush_size, h, w):
    """
    Tô 1 vùng boolean (H,W) từ đường đi con trỏ chuột (brush path), bằng cách
    "dán" (stamp) 1 hình tròn bán kính brush_size/2 dọc theo đường đi, nối các
    điểm liên tiếp bằng đoạn thẳng để không bị đứt quãng khi chuột di chuyển nhanh.
    """
    if len(points) < 1:
        raise ValueError("Cần ít nhất 1 điểm để vẽ brush.")

    radius = max(1, int(round(max(1.0, float(brush_size)) / 2)))
    footprint = disk(radius).astype(bool)
    fh, fw = footprint.shape
    region = np.zeros((h, w), dtype=bool)

    def stamp(cx, cy):
        cx, cy = int(round(cx)), int(round(cy))
        y0, x0 = cy - radius, cx - radius
        y1, x1 = y0 + fh, x0 + fw
        fy0, fx0, fy1, fx1 = 0, 0, fh, fw
        if y0 < 0:
            fy0 = -y0; y0 = 0
        if x0 < 0:
            fx0 = -x0; x0 = 0
        if y1 > h:
            fy1 -= (y1 - h); y1 = h
        if x1 > w:
            fx1 -= (x1 - w); x1 = w
        if y0 >= y1 or x0 >= x1:
            return
        region[y0:y1, x0:x1] |= footprint[fy0:fy1, fx0:fx1]

    step = max(1, radius // 2)
    prev = None
    for px, py in points:
        if prev is not None:
            rr, cc = sk_line(int(round(prev[1])), int(round(prev[0])), int(round(py)), int(round(px)))
            for i in range(0, len(rr), step):
                stamp(cc[i], rr[i])
            stamp(cc[-1], rr[-1])
        else:
            stamp(px, py)
        prev = (px, py)

    return region


def mask_handle(raw_img, op, exclude_arrays, include_arrays):
    """
    Tính vùng mới (boolean) từ thao tác flood fill / khoanh tay trên ảnh raw,
    sau đó áp post-process: trừ theo exclude_arrays, chỉ giữ trong include_arrays.
    op = {"mode": "floodfill", "x":.., "y":.., "tolerance":..}
      hoặc {"mode": "manual", "points": [[x,y],...]}
      hoặc {"mode": "brush", "points": [[x,y],...], "brush_size": ..}
    """
    h, w = raw_img.shape
    mode = op.get("mode")

    if mode == "floodfill":
        x, y = int(round(op["x"])), int(round(op["y"]))
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Tọa độ ngoài phạm vi ảnh.")
        tolerance = float(op.get("tolerance", 10))
        region = flood(raw_img, (y, x), tolerance=tolerance)
    elif mode == "manual":
        points = op.get("points", [])
        if len(points) < 3:
            raise ValueError("Cần ít nhất 3 điểm để tạo vùng khoanh.")
        xs = np.clip(np.array([p[0] for p in points], dtype=np.float64), 0, w - 1)
        ys = np.clip(np.array([p[1] for p in points], dtype=np.float64), 0, h - 1)
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        region = np.zeros((h, w), dtype=bool)
        region[rr, cc] = True
    elif mode == "brush":
        region = brush_path_to_region(op.get("points", []), op.get("brush_size", 10), h, w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    region = region.copy()
    for ex in exclude_arrays:
        region[ex.astype(bool)] = False

    if include_arrays:
        keep = np.zeros((h, w), dtype=bool)
        for inc in include_arrays:
            keep |= inc.astype(bool)
        region &= keep

    return region

def mask_muxing(existing_mask, handled_region, mask_mode):
    """Trộn mask hiện có (uint8) với vùng vừa xử lý -> mask mới (uint8)."""
    existing_bool = existing_mask.astype(bool)
    if mask_mode == "subtract":
        out = existing_bool & (~handled_region)
    else:
        out = existing_bool | handled_region
    return out.astype(np.uint8)

def mask_threshold(raw_hu_img, hu_min, hu_max):
    """
    Lấy ảnh raw HU => tạo mask np binary image (uint8: 0 hoặc 1)
    khi giá trị pixel nằm trong khoảng [hu_min, hu_max].
    """
    # Tạo mask boolean thỏa mãn điều kiện HU
    mask = (raw_hu_img >= hu_min) & (raw_hu_img <= hu_max)
    # Chuyển về kiểu uint8 theo chuẩn của hệ thống
    return mask.astype(np.uint8)

def mask_open_close(temp_special_mask, op_key, op_value):
    iterations = int(op_value)
    mask_bool = temp_special_mask.astype(bool)
    footprint = disk(1)
    if op_key == "closing":
        for _ in range(iterations):
            mask_bool = erosion(mask_bool, footprint)
        for _ in range(iterations):
            mask_bool = dilation(mask_bool, footprint)
    else:
        for _ in range(iterations):
            mask_bool = dilation(mask_bool, footprint)
        for _ in range(iterations):
            mask_bool = erosion(mask_bool, footprint)
    return mask_bool.astype(np.uint8)

def require_volume_loaded():
    if STATE["raw_hu"] is None:
        return jsonify({"success": False, "error": "Chưa tải dữ liệu DICOM. Vui lòng chọn file zip trước."}), 400
    return None


def clamp_slice_idx(idx):
    return max(0, min(STATE["num_slices"] - 1, int(idx)))


def require_mask(mask_id):
    ms = STATE["mask_store"]
    if not ms or not ms.exists(mask_id):
        return jsonify({"success": False, "error": f"Mask '{mask_id}' không tồn tại."}), 400
    return None


# ----------------------------------------------------------------------------
# Routes: nạp dữ liệu / điều hướng slide / HU window
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có file được gửi lên."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "Tên file rỗng."}), 400
    if not f.filename.lower().endswith(".zip"):
        return jsonify({"success": False, "error": "Vui lòng chọn file .zip chứa các file DICOM."}), 400

    tmp_zip_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
    try:
        os.close(tmp_zip_fd)
        f.save(tmp_zip_path)

        raw_hu, datasets, pixel_spacing, wc, ww = load_dicom_series_from_zip(tmp_zip_path)
        z, h, w = raw_hu.shape

        STATE["raw_hu"] = raw_hu
        STATE["datasets"] = datasets
        STATE["mask_store"] = MaskStore(z, h, w)
        STATE["num_slices"] = z
        STATE["height"] = h
        STATE["width"] = w
        STATE["pixel_spacing"] = pixel_spacing
        STATE["hu_min"] = wc - ww / 2.0
        STATE["hu_max"] = wc + ww / 2.0
        STATE["case_name"] = os.path.splitext(f.filename)[0]
        STATE["undo"] = {"mask_id": None, "slice_index": None, "mask_array": None}

        data_min = float(np.percentile(raw_hu, 0.1))
        data_max = float(np.percentile(raw_hu, 99.9))
        bound_lo = min(data_min, STATE["hu_min"] - 200, -1024.0)
        bound_hi = max(data_max, STATE["hu_max"] + 200, 1024.0)
        STATE["hu_bounds"] = (round(bound_lo), round(bound_hi))

        # MaskStore.__init__ đã tự tạo sẵn 3 mask buildin rỗng -> chọn mask đầu tiên làm mặc định
        default_mask_id = STATE["mask_store"].list_mask_ids()[0]

        image_b64 = render_final_img(0)

        return jsonify({
            "success": True,
            "case_name": STATE["case_name"],
            "num_slices": z,
            "height": h,
            "width": w,
            "slice_index": 0,
            "image": image_b64,
            "hu_min": STATE["hu_min"],
            "hu_max": STATE["hu_max"],
            "hu_bound_min": STATE["hu_bounds"][0],
            "hu_bound_max": STATE["hu_bounds"][1],
            "pixel_spacing": pixel_spacing,
            "masks": STATE["mask_store"].to_summary(),
            "default_mask_id": default_mask_id,
            "slice_entries": STATE["mask_store"].list_slice_entries(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc DICOM: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


@app.route("/get_slice", methods=["POST"])
def get_slice():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    image_b64 = render_final_img(idx)
    return jsonify({"success": True, "slice_index": idx, "image": image_b64})


@app.route("/set_hu_window", methods=["POST"])
def set_hu_window():
    """Cập nhật cửa sổ HU (window level / width) rồi render lại ảnh."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    try:
        idx = clamp_slice_idx(data.get("slice_index", 0))
        hu_min = float(data.get("hu_min", STATE["hu_min"]))
        hu_max = float(data.get("hu_max", STATE["hu_max"]))
        if hu_min >= hu_max:
            return jsonify({"success": False, "error": "HU thấp phải nhỏ hơn HU cao."}), 400

        STATE["hu_min"] = hu_min
        STATE["hu_max"] = hu_max

        image_b64 = render_final_img(idx)
        return jsonify({
            "success": True, "slice_index": idx, "image": image_b64,
            "hu_min": hu_min, "hu_max": hu_max,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------------------------------------------------------------------
# Routes: đặt tên / xóa tên slide
# ----------------------------------------------------------------------------
@app.route("/rename_slide", methods=["POST"])
def rename_slide():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"success": False, "error": "Tên slide không được để trống."}), 400
    STATE["mask_store"].set_slide_name(idx, name)
    return jsonify({"success": True, "slice_entries": STATE["mask_store"].list_slice_entries()})


@app.route("/delete_slide_name", methods=["POST"])
def delete_slide_name():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    STATE["mask_store"].delete_slide_name(idx)
    return jsonify({"success": True, "slice_entries": STATE["mask_store"].list_slice_entries()})


# ----------------------------------------------------------------------------
# Routes: quản lý mask (CRUD, hiển thị, filter include/exclude)
# ----------------------------------------------------------------------------
@app.route("/get_mask_list", methods=["POST"])
def get_mask_list():
    err = require_volume_loaded()
    if err:
        return err
    return jsonify({"success": True, "masks": STATE["mask_store"].to_summary()})


@app.route("/get_slices_with_mask", methods=["POST"])
def get_slices_with_mask():
    err = require_volume_loaded()
    if err:
        return err
    return jsonify({"success": True, "slices": STATE["mask_store"].list_slices_with_mask()})


@app.route("/create_mask", methods=["POST"])
def create_mask():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True) or {}
    mid = STATE["mask_store"].create_mask(label=data.get("label"), color=data.get("color"))
    return jsonify({"success": True, "mask_id": mid, "masks": STATE["mask_store"].to_summary()})


@app.route("/delete_mask", methods=["POST"])
def delete_mask():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    ok = STATE["mask_store"].delete_mask(mask_id)
    if not ok:
        return jsonify({"success": False, "error": "Không thể xóa mask có sẵn (buildin)."}), 400
    idx = clamp_slice_idx(data.get("slice_index", 0))
    if STATE["undo"]["mask_id"] == mask_id:
        STATE["undo"] = {"mask_id": None, "slice_index": None, "mask_array": None}
    image_b64 = render_final_img(idx)
    return jsonify({
        "success": True, "masks": STATE["mask_store"].to_summary(),
        "slice_entries": STATE["mask_store"].list_slice_entries(),
        "image": image_b64, "slice_index": idx,
    })


@app.route("/rename_mask", methods=["POST"])
def rename_mask():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    STATE["mask_store"].rename_mask(mask_id, str(data.get("label", "")).strip() or mask_id)
    return jsonify({"success": True, "masks": STATE["mask_store"].to_summary()})


@app.route("/recolor_mask", methods=["POST"])
def recolor_mask():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    color = data.get("color", [255, 0, 0])
    STATE["mask_store"].recolor_mask(mask_id, color)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    image_b64 = render_final_img(idx)
    return jsonify({"success": True, "masks": STATE["mask_store"].to_summary(), "image": image_b64, "slice_index": idx})


@app.route("/toggle_mask_visibility", methods=["POST"])
def toggle_mask_visibility():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    STATE["mask_store"].set_visibility(mask_id, data.get("visible", True))
    idx = clamp_slice_idx(data.get("slice_index", 0))
    image_b64 = render_final_img(idx)
    return jsonify({"success": True, "masks": STATE["mask_store"].to_summary(), "image": image_b64, "slice_index": idx})


@app.route("/set_mask_filters", methods=["POST"])
def set_mask_filters():
    """Cập nhật include-list / exclude-list gắn theo 1 mask_id."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    STATE["mask_store"].set_filters(mask_id, include=data.get("include"), exclude=data.get("exclude"))
    return jsonify({"success": True, "masks": STATE["mask_store"].to_summary()})


# ----------------------------------------------------------------------------
# Routes: sửa mask (flood fill / khoanh tay) + undo
# ----------------------------------------------------------------------------
@app.route("/mask_modify", methods=["POST"])
def mask_modify():
    """
    Route thống nhất thay cho /floodfill và /manual_mask cũ.
    body: slice_index, mask_id, mode ('floodfill'|'manual'), mask_mode ('add'|'subtract'),
          tolerance, x, y (floodfill)  hoặc  points (manual)
    """
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    try:
        idx = clamp_slice_idx(data.get("slice_index", 0))
        mask_mode = data.get("mask_mode", "add")
        ms = STATE["mask_store"]

        existing = ms.get_mask(idx, mask_id).copy()
        # lưu undo (chỉ 1 bước, chỉ cho route này)
        STATE["undo"] = {"mask_id": mask_id, "slice_index": idx, "mask_array": existing.copy()}

        raw = render_raw_img(idx)
        filters = ms.get_filters(mask_id)
        exclude_arrays = [ms.get_mask(idx, m) for m in filters["exclude"] if ms.exists(m)]
        include_arrays = [ms.get_mask(idx, m) for m in filters["include"] if ms.exists(m)]

        op = {
            "mode": data.get("mode"),
            "x": data.get("x"), "y": data.get("y"),
            "tolerance": data.get("tolerance", 10),
            "points": data.get("points", []),
            "brush_size": data.get("brush_size", 10),
        }
        handled = mask_handle(raw, op, exclude_arrays, include_arrays)
        new_mask = mask_muxing(existing, handled, mask_mode)
        ms.set_mask(idx, mask_id, new_mask)

        image_b64 = render_final_img(idx)
        return jsonify({
            "success": True, "slice_index": idx, "image": image_b64,
            "region_pixels": int(handled.sum()),
            "slice_entries": ms.list_slice_entries(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/undo_mask", methods=["POST"])
def undo_mask():
    """Ctrl+Z: chỉ hoàn tác được thao tác /mask_modify gần nhất, đúng 1 lần."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    mask_id = data.get("mask_id")

    u = STATE["undo"]
    if u["mask_id"] is not None and u["mask_id"] == mask_id and u["slice_index"] == idx:
        STATE["mask_store"].set_mask(idx, mask_id, u["mask_array"])
        STATE["undo"] = {"mask_id": None, "slice_index": None, "mask_array": None}
        image_b64 = render_final_img(idx)
        return jsonify({"success": True, "slice_index": idx, "image": image_b64, "undone": True})

    return jsonify({"success": True, "undone": False, "message": "Không có thao tác nào để hoàn tác."})


# ----------------------------------------------------------------------------
# Routes: xóa / hình thái học (đều thao tác trên 1 mask_id cụ thể)
# ----------------------------------------------------------------------------
@app.route("/clear_mask", methods=["POST"])
def clear_mask():
    """Xóa TOÀN BỘ mask (mọi mask_id) của 1 slide."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    STATE["mask_store"].clear_slice(idx)
    image_b64 = render_final_img(idx)
    return jsonify({
        "success": True, "slice_index": idx, "image": image_b64,
        "slice_entries": STATE["mask_store"].list_slice_entries(),
    })


@app.route("/clear_color_mask", methods=["POST"])
def clear_color_mask():
    """Xóa mask CHỈ của 1 mask_id, trên slide hiện tại."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    idx = clamp_slice_idx(data.get("slice_index", 0))
    STATE["mask_store"].clear_mask_on_slice(idx, mask_id)
    image_b64 = render_final_img(idx)
    return jsonify({
        "success": True, "slice_index": idx, "image": image_b64,
        "slice_entries": STATE["mask_store"].list_slice_entries(),
    })


@app.route("/denoise_mask", methods=["POST"])
def denoise_mask():
    """Closing thủ công: dilation x lần rồi erosion x lần, trên mask_id ở slide hiện tại."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    try:
        idx = clamp_slice_idx(data.get("slice_index", 0))
        iterations = max(0, min(int(data.get("iterations", 1)), 50))
        ms = STATE["mask_store"]
        mask = ms.get_mask(idx, mask_id).astype(bool)

        footprint = disk(1)
        for _ in range(iterations):
            mask = dilation(mask, footprint)
        for _ in range(iterations):
            mask = erosion(mask, footprint)

        ms.set_mask(idx, mask_id, mask.astype(np.uint8))
        image_b64 = render_final_img(idx)
        return jsonify({"success": True, "slice_index": idx, "image": image_b64, "mask_pixels": int(mask.sum())})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/remove_small_noise", methods=["POST"])
def remove_small_noise():
    """Opening thủ công: erosion x lần rồi dilation x lần, trên mask_id ở slide hiện tại."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    mask_id = data.get("mask_id")
    err = require_mask(mask_id)
    if err:
        return err
    try:
        idx = clamp_slice_idx(data.get("slice_index", 0))
        iterations = max(0, min(int(data.get("iterations", 1)), 50))
        ms = STATE["mask_store"]
        mask = ms.get_mask(idx, mask_id).astype(bool)

        footprint = disk(1)
        for _ in range(iterations):
            mask = erosion(mask, footprint)
        for _ in range(iterations):
            mask = dilation(mask, footprint)

        ms.set_mask(idx, mask_id, mask.astype(np.uint8))
        image_b64 = render_final_img(idx)
        return jsonify({"success": True, "slice_index": idx, "image": image_b64, "mask_pixels": int(mask.sum())})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------------------------------------------------------------------
# Routes: thống kê / xuất DICOM-SEG
# ----------------------------------------------------------------------------
@app.route("/get_stats", methods=["POST"])
def get_stats():
    """Thống kê diện tích từng mask trên slide hiện tại + thông tin WL/WW/HU/slide."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))

    row_mm, col_mm = STATE["pixel_spacing"]
    px_area_mm2 = row_mm * col_mm
    ms = STATE["mask_store"]

    results = []
    for mid, info in ms.get_display_masks(idx).items():
        m = info["array"].astype(bool)
        total_px = int(m.sum())
        results.append({
            "id": mid,
            "label": info["label"],
            "color": list(info["color"]),
            "visible": info["visible"],
            "total_pixels": total_px,
            "total_area_mm2": round(total_px * px_area_mm2, 2),
            "total_area_cm2": round(total_px * px_area_mm2 / 100.0, 3),
        })

    hu_min, hu_max = STATE["hu_min"], STATE["hu_max"]
    wl = (hu_min + hu_max) / 2.0
    ww = hu_max - hu_min

    return jsonify({
        "success": True, "slice_index": idx, "stats": results,
        "hu_min": round(hu_min, 1), "hu_max": round(hu_max, 1),
        "wl": round(wl, 1), "ww": round(ww, 1),
        "num_slices": STATE["num_slices"],
        "case_name": STATE["case_name"],
    })


@app.route("/export_dicom_seg", methods=["POST"])
def export_dicom_seg():
    """Xuất DICOM-SEG với số lượng segment ĐỘNG theo số mask hiện có."""
    err = require_volume_loaded()
    if err:
        return err
    try:
        datasets = STATE["datasets"]
        ms = STATE["mask_store"]
        mask_ids = ms.list_mask_ids()
        z, h, w = STATE["num_slices"], STATE["height"], STATE["width"]

        if not mask_ids:
            return jsonify({"success": False, "error": "Chưa có mask nào để xuất."}), 400

        seg_array = np.zeros((z, h, w, len(mask_ids)), dtype=np.uint8)
        has_any = False
        for ch, mid in enumerate(mask_ids):
            for sl in range(z):
                arr = ms.get_mask(sl, mid)
                if arr.any():
                    has_any = True
                seg_array[sl, :, :, ch] = arr

        if not has_any:
            return jsonify({"success": False, "error": "Chưa có mask nào có dữ liệu để xuất."}), 400
        try:
            import highdicom as hd
        except:
            return jsonify({"success": False, "error": "Chưa cài đặt thư viện highdicom. Hãy cài đặt Microsoft Visual C++ Build Tools."}), 400
        segment_descriptions = []
        for i, mid in enumerate(mask_ids):
            label_text = ms.meta[mid]["label"] or mid
            segment_descriptions.append(
                hd.seg.SegmentDescription(
                    segment_number=i + 1,
                    segment_label=label_text,
                    segmented_property_category=Code("91723000", "SCT", "Anatomical structure"),
                    segmented_property_type=Code("91723000", "SCT", "Anatomical structure"),
                    algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
                )
            )

        seg_dataset = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=seg_array,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_instance_uid=hd.UID(),
            series_number=999,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="CT-Segmentation-App",
            manufacturer_model_name="FlaskSegViewer",
            software_versions="2.0",
            device_serial_number="0001",
            content_label="MANUAL_SEG",
            content_description="Segmentation nhieu mask dong tao bang cong cu web",
        )

        # Lưu thêm metadata (tên slide + status của từng mask_id) vào private tag,
        # best-effort: nếu vì lý do gì đó không gắn được thì vẫn xuất file bình thường.
        try:
            extra_meta = {
                "slide_names": {str(k): v for k, v in ms.slide_names.items()},
                "mask_status": {mid: ms.meta[mid].get("status", "custom") for mid in mask_ids},
                "mask_order": mask_ids,
            }
            seg_dataset.add_new((0x0041, 0x0010), "LO", "CTSEGAPP")
            seg_dataset.add_new((0x0041, 0x1001), "LT", json.dumps(extra_meta)[:10240])
        except Exception:
            pass

        buf = io.BytesIO()
        seg_dataset.save_as(buf)
        buf.seek(0)

        return send_file(
            buf, mimetype="application/dicom", as_attachment=True,
            download_name=f"{STATE['case_name'] or 'segmentation'}.dcm",
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi xuất DICOM-SEG: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
