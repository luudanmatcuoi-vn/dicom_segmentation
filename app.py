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

DICOM-SEG (chuẩn):
- Upload zip: server LUÔN tạo 3 mask custom rỗng, rồi tìm file DICOM-SEG (SOP Class 1.2.840.10008.5.1.4.1.1.66.4).
  Mỗi Segment: trùng tên với mask custom có sẵn -> ghi đè pixel + đổi màu theo SEG; không trùng -> tạo mask custom mới,
  pixel được ánh xạ về đúng slice qua ReferencedSOPInstanceUID (dự phòng: ImagePositionPatient).
- Export: trả về 1 file ZIP = TOÀN BỘ nội dung zip gốc (giữ nguyên byte) + 1 file DICOM-SEG mới
  chứa các mask "custom" có dữ liệu (Segment Number 1..N, Segment Label = tên mask).
  Các file SEG cũ đã được nạp vào state sẽ được thay bằng file SEG mới (tránh trùng lặp).
  Không có mask custom nào có dữ liệu -> zip DICOM thường (không có SEG).

Route mới:
- /get_slice_nomask (POST) : ảnh của slide KHÔNG có mask (FE dùng cho nút "giữ để ẩn mask").
- /get_hu_data      (GET)  : HU của 1 slide dạng nhị phân int16 little-endian, xem docstring của route.

Tương thích OHIF Viewer (2 chiều, xem build_dicom_seg_bytes / import_seg_items):
- Export: SEG dùng highdicom, ReferencedSeriesSequence/FrameOfReferenceUID lấy tự động từ
  source_images nên OHIF (cornerstone-dicom-seg) đọc trực tiếp được khi mở cùng zip CT gốc.
  omit_empty_frames=True để file nhẹ hơn; content_qualification=RESEARCH (nếu highdicom hỗ
  trợ) để đánh dấu không phải kết quả lâm sàng chính thức. SegmentsOverlap để highdicom tự
  suy ra từ dữ liệu thật (mask của tool có thể chồng pixel nhau).
- Import: _is_seg_dataset nhận diện SEG qua SOPClassUID/Modality, dự phòng thêm bằng
  SegmentSequence cho các file SEG thiếu field chuẩn (kể cả SEG do OHIF hoặc phần mềm khác
  xuất ra, không chỉ SEG do chính tool này tạo).
"""
import io
import os
import json
import zipfile
import tempfile
import shutil
import base64
import atexit
import unicodedata
from urllib.parse import quote

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
        # LUÔN tạo 3 mask custom rỗng mặc định (nếu zip có SEG cùng tên thì import_seg_items sẽ ghi đè lên)
        self.add_default_custom_masks()

        # Tạo các mask buildin sẵn
        self.create_mask(label = "_cơ_xương", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[29-24,29+24]}, {"opening":2}])
        self.create_mask(label = "_mỡ_dưới_da", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[0-93-23,0-93+23]}, {"opening":4}])
        self.create_mask(label = "_mỡ_nội_tạng", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[0-100,0-50]}, {"opening":1}])
        self.create_mask(label = "_khí", color=DEFAULT_PALETTE[3], status="buildin", visible=False, special=[{"threshold":[-100000000,-200]}, {"opening":2}])

    def add_default_custom_masks(self):
        self.create_mask(label = "Cơ xương", color=DEFAULT_PALETTE[0], status="custom")
        self.create_mask(label = "Mỡ dưới da", color=DEFAULT_PALETTE[1], status="custom")
        self.create_mask(label = "Mỡ nội tạng", color=DEFAULT_PALETTE[2], status="custom")

    @staticmethod
    def _norm_label(label):
        return unicodedata.normalize("NFC", str(label or "")).strip().casefold()

    def find_custom_by_label(self, label, exclude=()):
        """Mask custom có tên trùng (không phân biệt hoa/thường, khoảng trắng đầu/cuối), chưa nằm trong exclude."""
        key = self._norm_label(label)
        for mid, m in self.meta.items():
            if m.get("status") == "custom" and mid not in exclude and self._norm_label(m["label"]) == key:
                return mid
        return None

    def clear_mask_pixels(self, mask_id):
        """Xóa dữ liệu pixel của 1 mask trên MỌI slide (giữ nguyên mask/tên/màu)."""
        for sl in self.pixels.values():
            sl.pop(mask_id, None)

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
    "source_zip_path": None,          # bản sao zip gốc trên đĩa (để export zip = zip gốc + SEG)
    "consumed_seg_entries": set(),    # tên entry SEG trong zip đã được nạp vào mask_store (sẽ bị thay khi export)
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


SEG_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"     # Segmentation Storage
PRIVATE_CREATOR = "CTSEGAPP"                              # private block (0041,10xx) lưu metadata riêng của app


def _is_junk_entry(name):
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts:
        return True
    return parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts)


def _is_seg_dataset(ds):
    if str(getattr(ds, "SOPClassUID", "")) == SEG_SOP_CLASS_UID:
        return True
    if str(getattr(ds, "Modality", "")).upper() == "SEG":
        return True
    # Tín hiệu phụ: có SegmentSequence (đặc trưng riêng của DICOM-SEG) nhưng
    # SOPClassUID/Modality bị thiếu hoặc không chuẩn (một số exporter bên thứ 3,
    # kể cả OHIF ở vài phiên bản cũ, có thể ghi thiếu field này).
    if getattr(ds, "SegmentSequence", None):
        return True
    return False


def load_dicom_zip(zip_path):
    """
    Đọc zip (không giải nén ra đĩa). Tách:
      - các slice ảnh (2D, có pixel data) -> raw_hu (Z,H,W) sắp theo vị trí z
      - các file DICOM-SEG -> seg_items [(tên entry trong zip, dataset)]
    """
    image_items = []     # (ds, pixel_array)
    seg_items = []       # (entry_name, ds)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or _is_junk_entry(name):
                continue
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(info)), force=True)
            except Exception:
                continue
            if _is_seg_dataset(ds):
                seg_items.append((name, ds))
                continue
            try:
                arr = ds.pixel_array
            except Exception:
                continue
            if arr.ndim != 2:          # bỏ multi-frame / ảnh màu
                continue
            image_items.append((ds, arr))

    if not image_items:
        raise ValueError("Không tìm thấy file DICOM ảnh hợp lệ trong zip.")

    def sort_key(item):
        ds = item[0]
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) == 3:
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return 0.0

    image_items.sort(key=sort_key)

    first = image_items[0][0]
    wc = _get_first(getattr(first, "WindowCenter", None), 40.0)
    ww = _get_first(getattr(first, "WindowWidth", None), 400.0)

    ps = getattr(first, "PixelSpacing", None)
    if ps and len(ps) == 2:
        pixel_spacing = (float(ps[0]), float(ps[1]))
    else:
        pixel_spacing = (1.0, 1.0)

    h0, w0 = image_items[0][1].shape
    hu_slices, kept_datasets = [], []
    for ds, arr in image_items:
        if arr.shape != (h0, w0):
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu_slices.append(arr.astype(np.float32) * slope + intercept)
        kept_datasets.append(ds)

    return {
        "raw_hu": np.stack(hu_slices, axis=0),
        "datasets": kept_datasets,
        "pixel_spacing": pixel_spacing,
        "wc": wc, "ww": ww,
        "seg_items": seg_items,
    }


# ----------------------------------------------------------------------------
# Màu segment: RGB <-> CIELab (DICOM RecommendedDisplayCIELabValue, illuminant D50)
# ----------------------------------------------------------------------------
_D50 = np.array([0.96422, 1.0, 0.82521])
_SRGB_TO_XYZ_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                             [0.2225045, 0.7168786, 0.0606169],
                             [0.0139322, 0.0971045, 0.7141733]])
_XYZ_D50_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ_D50)


def cielab_dicom_to_rgb(v):
    """[L,a,b] mã hóa uint16 theo DICOM -> (r,g,b) 0..255."""
    L = v[0] * 100.0 / 65535.0
    a = v[1] * 255.0 / 65535.0 - 128.0
    b = v[2] * 255.0 / 65535.0 - 128.0
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    finv = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787
    xyz = _D50 * np.array([finv(fx), finv(fy), finv(fz)])
    lin = np.clip(_XYZ_D50_TO_SRGB @ xyz, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return tuple(int(round(float(c) * 255)) for c in np.clip(srgb, 0, 1))


def rgb_to_cielab_dicom(rgb):
    """(r,g,b) 0..255 -> [L,a,b] mã hóa uint16 theo DICOM."""
    c = np.array(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (_SRGB_TO_XYZ_D50 @ lin) / _D50
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L, a, b = 116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])
    enc = lambda x: int(round(min(max(x, 0.0), 65535.0)))
    return [enc(L / 100.0 * 65535.0), enc((a + 128.0) / 255.0 * 65535.0), enc((b + 128.0) / 255.0 * 65535.0)]


# ----------------------------------------------------------------------------
# Nạp DICOM-SEG -> MaskStore
# ----------------------------------------------------------------------------
def read_private_meta(ds):
    """Metadata riêng của app (tên slide, màu RGB chính xác) trong private block 0041,10xx."""
    try:
        creator = ds.get((0x0041, 0x0010))
        if creator is None or str(creator.value).strip() != PRIVATE_CREATOR:
            return {}
        elem = ds.get((0x0041, 0x1001))
        if elem is None:
            return {}
        v = elem.value
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        meta = json.loads(str(v).rstrip("\x00 "))
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _frame_to_slice_index(fg, sop_to_idx, ipps, tol):
    """Xác định frame SEG thuộc slice nào: ưu tiên ReferencedSOPInstanceUID, dự phòng ImagePositionPatient."""
    try:
        ref = str(fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        if ref in sop_to_idx:
            return sop_to_idx[ref]
    except Exception:
        pass
    try:
        ipp = np.array([float(x) for x in fg.PlanePositionSequence[0].ImagePositionPatient])
        d = np.linalg.norm(ipps - ipp, axis=1)
        j = int(np.argmin(d))
        if d[j] <= tol:
            return j
    except Exception:
        pass
    return None


def import_seg_items(seg_items, datasets, store):
    """
    Nạp mọi DICOM-SEG tìm được vào store (store đã có sẵn 3 mask custom rỗng mặc định).
    Với mỗi Segment:
      - có mask custom trùng tên (SegmentLabel) -> GHI ĐÈ pixel của mask đó bằng segment và đổi theo màu của SEG;
      - không trùng tên -> tạo mask custom mới.
    Màu SEG: private meta > RecommendedDisplayCIELabValue (không có màu thì giữ màu mask cũ / palette).
    Trả về dict: consumed (set tên entry đã nạp), warnings, n_segments, n_replaced, n_new, n_frames.
    """
    result = {"consumed": set(), "warnings": [], "n_segments": 0, "n_replaced": 0, "n_new": 0, "n_frames": 0}
    claimed = set()      # mask đã bị 1 segment chiếm -> segment trùng tên tiếp theo sẽ tạo mask mới, không ghi đè lần nữa
    if not seg_items:
        return result

    sop_to_idx = {str(ds.SOPInstanceUID): i for i, ds in enumerate(datasets) if hasattr(ds, "SOPInstanceUID")}
    ipps = np.full((len(datasets), 3), 1e18)
    for i, ds in enumerate(datasets):
        p = getattr(ds, "ImagePositionPatient", None)
        if p is not None and len(p) == 3:
            ipps[i] = [float(x) for x in p]
    zs = np.sort(ipps[ipps[:, 2] < 1e17][:, 2])
    diffs = np.abs(np.diff(zs)) if len(zs) > 1 else np.array([])
    diffs = diffs[diffs > 1e-6]
    tol = 0.5 * float(np.median(diffs)) if len(diffs) else 0.5

    for name, sds in seg_items:
        short = os.path.basename(name)
        created = []         # mask tạo mới
        reused = []          # (mask_id, màu cũ) của mask mặc định bị ghi đè

        def rollback():
            for m in created:
                store.delete_mask(m)
            for m, old_color in reused:
                store.recolor_mask(m, old_color)
                store.clear_mask_pixels(m)
                claimed.discard(m)

        try:
            if (int(sds.Rows), int(sds.Columns)) != (store.h, store.w):
                result["warnings"].append(f"{short}: kích thước SEG khác ảnh, bỏ qua.")
                continue
            seg_seq = getattr(sds, "SegmentSequence", None)
            if not seg_seq:
                result["warnings"].append(f"{short}: không có SegmentSequence, bỏ qua.")
                continue

            priv = read_private_meta(sds)
            priv_segments = priv.get("segments", {}) if isinstance(priv.get("segments"), dict) else {}

            # 1) tạo mask cho từng Segment (theo SegmentNumber tăng dần)
            num_to_mid = {}
            for seg in sorted(seg_seq, key=lambda s: int(s.SegmentNumber)):
                sn = int(seg.SegmentNumber)
                label = str(getattr(seg, "SegmentLabel", "") or "").strip() or f"Segment {sn}"
                color = None
                pc = priv_segments.get(str(sn), {}).get("color")
                if isinstance(pc, (list, tuple)) and len(pc) == 3:
                    color = tuple(int(c) for c in pc)
                if color is None:
                    lab = getattr(seg, "RecommendedDisplayCIELabValue", None)
                    if lab is not None and len(lab) == 3:
                        color = cielab_dicom_to_rgb([int(x) for x in lab])
                mid = store.find_custom_by_label(label, exclude=claimed)
                if mid is not None:                       # trùng tên -> thay mask cũ, đổi màu theo SEG
                    claimed.add(mid)
                    reused.append((mid, store.meta[mid]["color"]))
                    store.clear_mask_pixels(mid)
                    if color is not None:
                        store.recolor_mask(mid, color)
                else:
                    mid = store.create_mask(label=label, color=color, status="custom")
                    created.append(mid)
                num_to_mid[sn] = mid

            # 2) đọc frame -> slice
            frames = sds.pixel_array
            if frames.ndim == 2:
                frames = frames[None]
            pf = getattr(sds, "PerFrameFunctionalGroupsSequence", None) or []
            fractional = str(getattr(sds, "SegmentationType", "BINARY")).upper() == "FRACTIONAL"
            thr = float(getattr(sds, "MaximumFractionalValue", 255)) / 2.0 if fractional else 0.0

            mapped = unmapped = 0
            for f in range(min(frames.shape[0], len(pf))):
                fg = pf[f]
                try:
                    sn = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                except Exception:
                    sn = next(iter(num_to_mid)) if len(num_to_mid) == 1 else None
                idx = _frame_to_slice_index(fg, sop_to_idx, ipps, tol)
                if sn not in num_to_mid or idx is None:
                    unmapped += 1
                    continue
                mid = num_to_mid[sn]
                binary = (frames[f] > thr)
                store.set_mask(idx, mid, store.get_mask(idx, mid).astype(bool) | binary)
                mapped += 1

            if mapped == 0:      # SEG này không thuộc series đang mở -> hoàn tác, không nạp
                rollback()
                result["warnings"].append(f"{short}: không khớp slice nào của series, bỏ qua.")
                continue
            if unmapped:
                result["warnings"].append(f"{short}: {unmapped} frame không khớp slice nào.")

            # 3) tên slide (metadata riêng của app)
            for k, v in (priv.get("slide_names") or {}).items():
                try:
                    if 0 <= int(k) < store.num_slices:
                        store.set_slide_name(int(k), str(v))
                except Exception:
                    pass

            result["consumed"].add(name)
            result["n_segments"] += len(created) + len(reused)
            result["n_replaced"] += len(reused)
            result["n_new"] += len(created)
            result["n_frames"] += mapped
        except Exception as e:
            rollback()
            result["warnings"].append(f"{short}: lỗi đọc SEG ({e}).")
    return result


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


def render_final_img(slice_idx, with_masks=True):
    """Ảnh raw (+ trộn màu các mask đang visible nếu with_masks) -> base64 PNG."""
    gray = render_raw_img(slice_idx)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float64)

    if with_masks:
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
    keep_zip = False
    try:
        os.close(tmp_zip_fd)
        f.save(tmp_zip_path)

        loaded = load_dicom_zip(tmp_zip_path)
        raw_hu, datasets = loaded["raw_hu"], loaded["datasets"]
        pixel_spacing, wc, ww = loaded["pixel_spacing"], loaded["wc"], loaded["ww"]
        z, h, w = raw_hu.shape

        STATE["raw_hu"] = raw_hu
        STATE["datasets"] = datasets
        STATE["num_slices"] = z
        STATE["height"] = h
        STATE["width"] = w
        STATE["pixel_spacing"] = pixel_spacing
        STATE["hu_min"] = wc - ww / 2.0
        STATE["hu_max"] = wc + ww / 2.0
        STATE["case_name"] = os.path.splitext(f.filename)[0]
        STATE["undo"] = {"mask_id": None, "slice_index": None, "mask_array": None}

        # ---- mask: nếu zip có DICOM-SEG thì nạp segment/mask/tên slide vào state ----
        store = MaskStore(z, h, w)               # luôn có 3 mask custom rỗng
        seg_info = import_seg_items(loaded["seg_items"], datasets, store)
        STATE["mask_store"] = store
        STATE["consumed_seg_entries"] = seg_info["consumed"]

        # giữ bản sao zip gốc để export (zip gốc + SEG mới)
        old_zip = STATE.get("source_zip_path")
        STATE["source_zip_path"] = tmp_zip_path
        keep_zip = True
        if old_zip and old_zip != tmp_zip_path and os.path.exists(old_zip):
            try:
                os.remove(old_zip)
            except OSError:
                pass

        data_min = float(np.percentile(raw_hu, 0.1))
        data_max = float(np.percentile(raw_hu, 99.9))
        bound_lo = min(data_min, STATE["hu_min"] - 200, -1024.0)
        bound_hi = max(data_max, STATE["hu_max"] + 200, 1024.0)
        STATE["hu_bounds"] = (round(bound_lo), round(bound_hi))

        default_mask_id = store.list_mask_ids()[0]

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
            "masks": store.to_summary(),
            "default_mask_id": default_mask_id,
            "slice_entries": store.list_slice_entries(),
            "seg_loaded": bool(seg_info["consumed"]),
            "seg_segments": seg_info["n_segments"],
            "seg_replaced": seg_info["n_replaced"],
            "seg_new": seg_info["n_new"],
            "seg_frames": seg_info["n_frames"],
            "seg_warnings": seg_info["warnings"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi đọc DICOM: {str(e)}"}), 500
    finally:
        if not keep_zip and os.path.exists(tmp_zip_path):
            os.remove(tmp_zip_path)


@atexit.register
def _cleanup_source_zip():
    p = STATE.get("source_zip_path")
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


@app.route("/get_slice", methods=["POST"])
def get_slice():
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    image_b64 = render_final_img(idx)
    return jsonify({"success": True, "slice_index": idx, "image": image_b64})


@app.route("/get_slice_nomask", methods=["POST"])
def get_slice_nomask():
    """Ảnh của slide KHÔNG có mask (theo cửa sổ HU hiện tại). FE dùng cho nút 'giữ để ẩn mask'."""
    err = require_volume_loaded()
    if err:
        return err
    data = request.get_json(force=True)
    idx = clamp_slice_idx(data.get("slice_index", 0))
    return jsonify({"success": True, "slice_index": idx, "image": render_final_img(idx, with_masks=False)})


@app.route("/get_hu_data", methods=["GET"])
def get_hu_data():
    """
    HU của 1 slide, GET /get_hu_data?slice_index=N

    Định dạng: body nhị phân thô (application/octet-stream), H*W giá trị int16 LITTLE-ENDIAN,
    theo thứ tự hàng (row-major): phần tử thứ (y*W + x) là HU của pixel (x, y).
    Kích thước nằm trong header: X-Width, X-Height, X-Slice-Index (ảnh 512x512 ~ 512 KB).

      - JS   : new Int16Array(await res.arrayBuffer())[y * W + x]
      - Flask: np.frombuffer(body, dtype="<i2").reshape(H, W)
    HU được làm tròn về số nguyên và kẹp trong [-32768, 32767].
    """
    err = require_volume_loaded()
    if err:
        return err
    try:
        idx = clamp_slice_idx(request.args.get("slice_index", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "slice_index không hợp lệ."}), 400
    hu = np.clip(np.rint(STATE["raw_hu"][idx]), -32768, 32767).astype("<i2")
    resp = app.response_class(hu.tobytes(), mimetype="application/octet-stream")
    resp.headers["X-Width"] = str(STATE["width"])
    resp.headers["X-Height"] = str(STATE["height"])
    resp.headers["X-Slice-Index"] = str(idx)
    resp.headers["Cache-Control"] = "no-store"
    return resp


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


def build_dicom_seg_bytes(ms, datasets, export_ids):
    """
    Tạo 1 file DICOM-SEG (bytes) chứa các mask trong export_ids.
    Segment Number = 1..N theo thứ tự export_ids, Segment Label = tên mask (tối đa 64 ký tự),
    màu hiển thị = RecommendedDisplayCIELabValue (chuẩn DICOM) + RGB chính xác trong private tag của app.
    Raise ImportError nếu chưa cài highdicom.
    """
    import highdicom as hd

    z, h, w = STATE["num_slices"], STATE["height"], STATE["width"]
    ch_of = {mid: i for i, mid in enumerate(export_ids)}
    seg_array = np.zeros((z, h, w, len(export_ids)), dtype=np.uint8)
    for sl_idx, d in ms.pixels.items():
        for mid, arr in d.items():
            ch = ch_of.get(mid)
            if ch is not None:
                seg_array[sl_idx, :, :, ch] = arr

    # content_qualification là tham số tùy chọn tùy phiên bản highdicom; nếu bản
    # cài đặt không có enum này thì bỏ qua để không làm hỏng luồng export hiện có.
    extra_seg_kwargs = {}
    if hasattr(hd, "ContentQualificationValues"):
        extra_seg_kwargs["content_qualification"] = hd.ContentQualificationValues.RESEARCH

    category = Code("85756007", "SCT", "Tissue")
    segment_descriptions, priv_segments = [], {}
    for i, mid in enumerate(export_ids):
        meta = ms.meta[mid]
        number = i + 1
        label = (meta["label"] or "").strip()[:64] or f"Segment {number}"
        desc = hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=label,
            segmented_property_category=category,
            segmented_property_type=category,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
        )
        desc.RecommendedDisplayCIELabValue = rgb_to_cielab_dicom(meta["color"])
        segment_descriptions.append(desc)
        priv_segments[str(number)] = {"label": label, "color": [int(c) for c in meta["color"]]}

    series_numbers = [int(ds.SeriesNumber) for ds in datasets if getattr(ds, "SeriesNumber", None) not in (None, "")]
    seg_dataset = hd.seg.Segmentation(
        source_images=datasets,
        pixel_array=seg_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=max(series_numbers, default=0) + 1,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="CT-Segmentation-App",
        manufacturer_model_name="FlaskSegViewer",
        software_versions="3.0",
        device_serial_number="0001",
        series_description=f"Segmentation ({len(export_ids)} segments)",
        content_label="MANUAL_SEG",
        content_description="Segmentation tao bang cong cu web",
        # Các tham số dưới đây ghi rõ (thay vì để mặc định ngầm) để file SEG tương
        # thích ổn định với OHIF (cornerstone-dicom-seg loader):
        omit_empty_frames=True,                 # bỏ frame toàn 0 -> file nhẹ hơn, OHIF đọc nhanh hơn
        # KHÔNG tự set SegmentsOverlap: highdicom tự tính từ seg_array thực tế
        # (mask của tool có thể chồng pixel giữa các mask -> để lib tự phát hiện là đúng nhất).
        **extra_seg_kwargs,
    )
    # nhãn tiếng Việt -> UTF-8
    seg_dataset.SpecificCharacterSet = "ISO_IR 192"

    # metadata riêng (tên slide + RGB chính xác) trong private block; viewer khác sẽ bỏ qua an toàn
    extra_meta = {
        "app": PRIVATE_CREATOR, "version": 3,
        "slide_names": {str(k): v for k, v in ms.slide_names.items() if 0 <= int(k) < z},
        "segments": priv_segments,
    }
    seg_dataset.add_new((0x0041, 0x0010), "LO", PRIVATE_CREATOR)
    seg_dataset.add_new((0x0041, 0x1001), "UT", json.dumps(extra_meta))

    buf = io.BytesIO()
    seg_dataset.save_as(buf)
    return buf.getvalue(), str(seg_dataset.SOPInstanceUID)


@app.route("/export_dicom_seg", methods=["POST"])
def export_dicom_seg():
    """
    Xuất file ZIP = toàn bộ nội dung zip đã upload (giữ nguyên) + 1 file DICOM-SEG mới.
    - Chỉ đưa vào SEG các mask 'custom' CÓ dữ liệu (mask buildin không xuất; mask custom rỗng bị bỏ qua).
    - Không có mask nào như vậy -> zip DICOM thường (không thêm SEG).
    - Các file SEG cũ đã nạp vào state được thay bằng SEG mới (state là bản mới nhất).
    Header phản hồi: X-Download-Name (url-encoded), X-Segments-Exported (số segment).
    """
    err = require_volume_loaded()
    if err:
        return err
    src_zip = STATE.get("source_zip_path")
    if not src_zip or not os.path.exists(src_zip):
        return jsonify({"success": False, "error": "Không còn file zip gốc trên server. Hãy upload lại."}), 400

    out_path = None
    try:
        ms = STATE["mask_store"]
        datasets = STATE["datasets"]

        custom_ids = [mid for mid in ms.list_mask_ids() if ms.meta[mid].get("status") == "custom"]
        used = set()
        for d in ms.pixels.values():
            for mid in custom_ids:
                arr = d.get(mid)
                if arr is not None and arr.any():
                    used.add(mid)
        export_ids = [mid for mid in custom_ids if mid in used]

        seg_bytes, seg_uid = None, None
        if export_ids:
            try:
                seg_bytes, seg_uid = build_dicom_seg_bytes(ms, datasets, export_ids)
            except ImportError:
                return jsonify({"success": False, "error": "Chưa cài đặt thư viện highdicom (pip install highdicom). Trên Windows có thể cần Microsoft Visual C++ Build Tools."}), 400

        fd, out_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        consumed = STATE.get("consumed_seg_entries") or set()
        with zipfile.ZipFile(src_zip, "r") as zin, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            names = set()
            for info in zin.infolist():
                if info.filename in consumed:      # SEG cũ đã nạp -> thay bằng SEG mới (hoặc bỏ nếu người dùng đã xóa hết mask)
                    continue
                names.add(info.filename)
                zi = zipfile.ZipInfo(info.filename, info.date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = info.external_attr
                zout.writestr(zi, b"" if info.is_dir() else zin.read(info))
            if seg_bytes is not None:
                seg_name = f"SEG_{seg_uid.split('.')[-1][-8:] or 'segmentation'}.dcm"
                while seg_name in names:
                    seg_name = "_" + seg_name
                zi = zipfile.ZipInfo(seg_name)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zout.writestr(zi, seg_bytes)

        base = STATE["case_name"] or "dicom"
        download_name = f"{base}_seg.zip" if seg_bytes is not None else f"{base}.zip"
        resp = send_file(out_path, mimetype="application/zip", as_attachment=True, download_name=download_name)
        resp.headers["X-Download-Name"] = quote(download_name)
        resp.headers["X-Segments-Exported"] = str(len(export_ids))
        _path = out_path

        def _rm():
            try:
                os.remove(_path)
            except OSError:
                pass
        resp.call_on_close(_rm)
        out_path = None      # đã giao cho call_on_close dọn
        return resp
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi xuất DICOM: {str(e)}"}), 500
    finally:
        if out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
