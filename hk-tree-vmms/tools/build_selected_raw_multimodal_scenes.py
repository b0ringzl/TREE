"""Build an auditable 8-scene panorama + raw LiDAR evidence package.

Only spatial/temporal cropping, coordinate transforms and deterministic
downsampling are allowed.  No CEDD mask, SegmentAnyTree result, denoising,
ground removal, semantic filtering or clustering is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
# Reuse the project's INS convention instead of redefining Euler angles here.
from diagnose_lidar_camera_axes import world_from_flu

TREE_ROOT = PROJECT.parent
VMMS = TREE_ROOT.parent / "vmms"
SUSTECH_SCENE = (
    TREE_ROOT.parent
    / "TREE_SUSTECHPOINTS_SANDBOX/vendor/SUSTechPOINTS/data"
    / "VMMS_HWT-CEDDRAW-ANNOTATED-ALL-PM3S-SPARSE-NOMASK"
)
OUTPUT = TREE_ROOT / "交付/人工标注_原始LiDAR配准场景_20260907_v3_指定树种"
COORDINATES = PROJECT / "outputs/coordinates/frame_coordinates.csv"
TST_DRAFTS = PROJECT / "derived/jianshazui_frame_labeler/runtime/frame_drafts.json"
HWT_RECORDS = PROJECT / "derived/hewentian_frame_labeler/annotations/records"
TST_MAP = PROJECT / "derived/hk_lidar_pairs_pilot_20260907/map_cache/20250210-jianshazui.npy"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")

TARGET_POINTS = 150_000
SCENE_RADIUS_M = 45.0
IMAGE_SIZE = (2048, 1024)

SELECTIONS = [
    ("TST", "jianshazui_pano_1__001604", "红花羊蹄甲与榕树"),
    ("TST", "jianshazui_pano_1__001632", "红花羊蹄甲"),
    ("TST", "jianshazui_pano_2__003089", "榕树"),
    ("SB", "stubbs_pano_0__000681", "石栗"),
    ("SB", "stubbs_pano_1__001557", "榕树"),
    ("TST", "jianshazui_pano_1__002749", "狐尾椰子"),
    ("TST", "jianshazui_pano_2__001485", "狐尾椰子"),
    ("SB", "stubbs_pano_1__002966", "榕树"),
]
ROUTE_NAMES = {'HWT': '何文田', 'TST': '尖沙咀', 'SB': '司徒拔道'}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def read_coordinates() -> dict[str, dict[str, str]]:
    with COORDINATES.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {f"{row['stream_id']}__{row['frame_id']}": row for row in rows}


def annotations() -> dict[str, dict]:
    result: dict[str, dict] = {}
    drafts = load_json(TST_DRAFTS)
    for key, value in drafts.items():
        if value.get("labels"):
            result[key] = value
    stubbs = PROJECT / 'derived/stubbs_road_frame_labeler'
    for key, value in load_json(stubbs / 'runtime/frame_drafts.json').items():
        if value.get('labels'):
            result[key] = value
    for path in (stubbs / 'annotations/records').glob('*.json'):
        value = load_json(path)
        if value.get('labels'):
            result[str(value.get('frame_key', value.get('frame_id', path.stem)))] = value
    hwt_drafts = load_json(HWT_RECORDS.parent.parent / 'runtime/frame_drafts.json')
    for key, value in hwt_drafts.items():
        if value.get('labels'):
            frame_id = str(value.get('frame_id', key)).zfill(6)
            result[f'hewentian_pano__{frame_id}'] = value
    for path in HWT_RECORDS.glob("*.json"):
        value = load_json(path)
        if value.get("labels"):
            result[f"hewentian_pano__{str(value.get('frame_id', path.stem)).zfill(6)}"] = value
    return result


def read_binary_pcd_xyz(path: Path) -> np.ndarray:
    header: list[str] = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"PCD header incomplete: {path}")
            text = line.decode("ascii", errors="strict").strip()
            header.append(text)
            if text.startswith("DATA "):
                data_type = text.split()[1]
                break
        meta = {line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1] for line in header if " " in line}
        fields = meta["FIELDS"].split()
        sizes = [int(x) for x in meta["SIZE"].split()]
        types = meta["TYPE"].split()
        counts = [int(x) for x in meta.get("COUNT", " ".join("1" for _ in fields)).split()]
        points = int(meta["POINTS"])
        if data_type != "binary" or any(size != 4 for size in sizes) or any(count != 1 for count in counts):
            raise RuntimeError(f"Unsupported PCD layout: {path}")
        dtype_fields = []
        for name, kind in zip(fields, types):
            dtype_fields.append((name, "<f4" if kind == "F" else "<u4" if kind == "U" else "<i4"))
        array = np.frombuffer(stream.read(), dtype=np.dtype(dtype_fields), count=points)
    return np.column_stack([array[name] for name in ("x", "y", "z")]).astype(np.float64)


def write_ascii_pcd(path: Path, xyz: np.ndarray) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA ascii\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        np.savetxt(stream, xyz.astype(np.float32), fmt="%.6f %.6f %.6f")


def world_to_camera(xyz: np.ndarray, row: dict[str, str], yaw_deg: float = 180.0) -> np.ndarray:
    position = np.array([float(row[f"local_{axis}"]) for axis in "xyz"])
    rotation = world_from_flu(*[float(row[f"{axis}_rad"]) for axis in ("roll", "pitch", "heading")])
    angle = math.radians(yaw_deg)
    offset = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    return (xyz - position) @ (rotation @ offset)


def sample_exact(xyz: np.ndarray, key: str) -> np.ndarray:
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) <= TARGET_POINTS:
        return xyz.copy()
    ids = np.random.default_rng(stable_seed(key)).choice(len(xyz), TARGET_POINTS, replace=False)
    return xyz[np.sort(ids)]


def project_camera(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(xyz, axis=1)
    valid = distance > 1e-6
    points = xyz[valid]
    distance = distance[valid]
    u = (0.5 - np.arctan2(points[:, 1], points[:, 0]) / (2 * np.pi)) % 1.0
    v = 0.5 - np.arcsin(np.clip(points[:, 2] / distance, -1, 1)) / np.pi
    uv = np.column_stack((u, v))
    good = (v >= 0) & (v <= 1)
    return uv[good], distance[good]


def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.exists() else ImageFont.load_default()


def annotated_panorama(source: Path, annotation: dict, out: Path) -> Image.Image:
    image = Image.open(source).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    colors = [(255, 196, 37, 255), (255, 92, 92, 255), (24, 229, 174, 255), (90, 160, 255, 255)]
    labels = annotation.get("labels") or []
    for index, label in enumerate(labels, start=1):
        points = [(float(x) * image.width, float(y) * image.height) for x, y in label.get("points", [])]
        if len(points) < 3:
            continue
        color = colors[(index - 1) % len(colors)]
        draw.polygon(points, fill=(color[0], color[1], color[2], 28))
        draw.line(points + [points[0]], fill=color, width=5)
        x, y = points[0]
        species = str(label.get("species", "未命名"))
        text = f"树{index}"
        x = min(max(x, 5), image.width - 120)
        y = min(max(y, 60), image.height - 45)
        box = draw.textbbox((x, y), text, font=font(30))
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(4, 18, 30, 210))
        draw.text((x, y), text, font=font(30), fill=color)
    image.save(out, quality=94)
    return image


def registration_image(base: Image.Image, xyz: np.ndarray, out: Path) -> dict:
    uv, distance = project_camera(xyz)
    canvas = base.copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if len(distance):
        near, far = np.percentile(distance, [5, 95])
        scale = np.clip((distance - near) / max(far - near, 1e-6), 0, 1)
        # Only the closest point at each pixel is drawn. This is a rendering
        # z-buffer, not a change to the exported point cloud.
        pixels = np.floor(uv * np.array(canvas.size)).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, canvas.width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, canvas.height - 1)
        linear = pixels[:, 1] * canvas.width + pixels[:, 0]
        ordered = np.lexsort((distance, linear))
        order = ordered[np.r_[True, np.diff(linear[ordered]) != 0]]
        order = order[np.argsort(distance[order])[::-1]]
        for idx in order:
            u, v = uv[idx]
            t = float(scale[idx])
            color = (int(30 + 225 * t), int(235 - 80 * t), int(255 - 220 * t), 165)
            x, y = int(u * canvas.width), int(v * canvas.height)
            draw.point((x, y), fill=color)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(canvas, "RGBA")
    title = f"场景投影 | {len(xyz):,} 点 | INS航向约定已修正 | 近青远黄"
    d.rectangle((0, 0, canvas.width, 54), fill=(3, 14, 25, 210))
    d.text((18, 10), title, font=font(28), fill=(245, 249, 255, 255))
    canvas.save(out, quality=94)
    return {"projected_point_count": int(len(uv)), "distance_p05_m": float(np.percentile(distance, 5)), "distance_p95_m": float(np.percentile(distance, 95))}


def copy_original(source: Path, out: Path) -> None:
    shutil.copy2(source, out)


def hwt_points(frame_id: str) -> tuple[np.ndarray, dict]:
    path = SUSTECH_SCENE / f"cedd_review/pcd/{frame_id}.pcd"
    xyz = read_binary_pcd_xyz(path)
    distance = np.linalg.norm(xyz, axis=1)
    cropped = xyz[(distance >= 1.5) & (distance <= SCENE_RADIUS_M)]
    index = load_json(SUSTECH_SCENE / "cedd_review/index.json")
    entries = index.get("frames", index if isinstance(index, list) else [])
    entry = next(item for item in entries if str(item.get("frame_id", "")).zfill(6) == frame_id)
    info = {
        "matching_method": "strict temporal accumulation",
        "temporal_window_s": [-3.0, 3.0],
        "scan_count": entry.get("scan_count"),
        "scan_start": entry.get("scan_start"),
        "scan_end": entry.get("scan_end"),
        "source_point_count_before_display_sampling": entry.get("retained_point_count"),
        "intermediate_no_mask_pcd": str(path),
        "intermediate_point_count": int(len(xyz)),
        "points_within_45m": int(len(cropped)),
        "source_raw_scan_directory": str(VMMS / "2023-10-27_hewentian/pointcloud/scans"),
        "prohibited_processing": [],
    }
    return cropped, info


def tst_points(row: dict[str, str]) -> tuple[np.ndarray, dict]:
    survey = Path(row['source_image_relpath']).parts[0]
    map_path = TST_MAP.parent / f'{survey}.npy'
    xyz = np.load(map_path, mmap_mode="r")
    position = np.array([float(row[f"local_{axis}"]) for axis in "xyz"])
    delta = np.asarray(xyz[:, :2], dtype=np.float64) - position[:2]
    ids = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= SCENE_RADIUS_M**2)
    world = np.asarray(xyz[ids], dtype=np.float64)
    camera = world_to_camera(world, row)
    sources = sorted((VMMS / survey / 'pointcloud/mapping').glob("pointcloud_DS_*.pcd"))
    if not sources:
        raise RuntimeError(f'Missing map source files for {survey}')
    info = {
        "matching_method": "camera-centred spatial crop from mapping cloud",
        "spatial_radius_m": SCENE_RADIUS_M,
        "world_point_count_within_45m": int(len(world)),
        "source_vendor_downsampled_mapping_files": [str(path) for path in sources],
        "map_cache": str(map_path),
        "source_survey": survey,
        "map_cache_processing": "0.15 m voxel downsampling only; no semantic or geometry filtering",
        "camera_yaw_offset_deg": 180.0,
        "INS_heading_convention": "clockwise from north; ENU-from-FLU = level @ roll @ pitch",
        "rotation_implementation": "diagnose_lidar_camera_axes.world_from_flu (shared project function)",
        "fine_extrinsic_status": "empirical axis/yaw relation; registration image is a visual candidate, not calibration ground truth",
        "prohibited_processing": [],
    }
    return camera, info


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_coordinates()
    ann = annotations()
    hwt_index = load_json(SUSTECH_SCENE / "cedd_review/index.json")
    manifest_rows: list[dict] = []
    contact: list[tuple[str, Path]] = []

    for number, (route, key, short_species) in enumerate(SELECTIONS, start=1):
        if key not in rows or key not in ann:
            raise RuntimeError(f"Missing coordinate or annotation: {key}")
        row, annotation = rows[key], ann[key]
        if len(annotation.get('labels', [])) < 2:
            raise RuntimeError(f'Multiple human labels required: {key}')
        stream_id, frame_id = key.rsplit("__", 1)
        prefix = f"{number:02d}_{route}_{stream_id.replace('_pano', '').replace('jianshazui_', 'P')}_{frame_id}_{short_species}"
        folder = OUTPUT / prefix
        point_dir, image_dir, registration_dir = folder / "点云数据", folder / "影像数据", folder / "配准图片"
        for directory in (point_dir, image_dir, registration_dir):
            directory.mkdir(parents=True, exist_ok=True)

        source_image = VMMS / row["source_image_relpath"]
        original_path = image_dir / "panorama_original.jpg"
        annotated_path = image_dir / "panorama_annotated.jpg"
        registration_path = registration_dir / "registration_overlay.jpg"
        pcd_path = point_dir / "lidar_scene_sparse.pcd"
        annotation_path = image_dir / "annotation.json"

        copy_original(source_image, original_path)
        annotated = annotated_panorama(source_image, annotation, annotated_path)
        write_json(annotation_path, annotation)

        if route == "HWT":
            raw_points, provenance = hwt_points(frame_id)
        else:
            raw_points, provenance = tst_points(row)
        sampled = sample_exact(raw_points, key)
        write_ascii_pcd(pcd_path, sampled)
        projection = registration_image(annotated, sampled, registration_path)

        species = [str(label.get("species", "")) for label in annotation.get("labels", [])]
        details = {
            "scene_number": number,
            "route": ROUTE_NAMES[route],
            "frame_key": key,
            "frame_id": frame_id,
            "stream_id": stream_id,
            "species": species,
            "human_label_count": len(species),
            "source_panorama": str(source_image),
            "source_panorama_sha256": sha256(source_image),
            "camera_local_xyz": [float(row[f"local_{axis}"]) for axis in "xyz"],
            "camera_attitude_rad": [float(row[f"{axis}_rad"]) for axis in ("roll", "pitch", "heading")],
            "utc_seconds": float(row["utc_seconds"]),
            "point_count_before_final_sampling": int(len(raw_points)),
            "final_point_count": int(len(sampled)),
            "downsampling": {
                "method": "deterministic uniform sampling without replacement",
                "target_points": TARGET_POINTS,
                "reference": "scene visualization cap; WHU 8192 points is for individual trees and is not applied to a street scene",
                "random_seed": stable_seed(key),
            },
            "allowed_operations": ["temporal/spatial crop", "coordinate transform", "downsampling"],
            "not_used": ["CEDD mask", "SegmentAnyTree", "ground removal", "denoising", "clustering", "semantic filtering"],
            "lidar_provenance": provenance,
            "projection_statistics": projection,
            "generated_sha256": {
                "lidar_scene_sparse.pcd": sha256(pcd_path),
                "panorama_original.jpg": sha256(original_path),
                "panorama_annotated.jpg": sha256(annotated_path),
                "registration_overlay.jpg": sha256(registration_path),
            },
        }
        write_json(folder / "场景说明.json", details)
        manifest_rows.append({
            "场景序号": number,
            "文件夹": prefix,
            "路段": details["route"],
            "影像流": stream_id,
            "帧号": frame_id,
            "人工标注数": len(species),
            "树种": " | ".join(species),
            "点数": len(sampled),
            "匹配方法": provenance["matching_method"],
            "配准说明": provenance.get("fine_extrinsic_status", "INS motion compensation + known panorama axis relation"),
        })
        contact.append((f"{number:02d} {details['route']} {frame_id} {short_species}", registration_path))
        print(f"built {number:02d}/08 {key}: {len(raw_points):,} -> {len(sampled):,}", flush=True)

    with (OUTPUT / "总清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    sheet = Image.new("RGB", (1600, 4 * 450), "#08131f")
    for i, (title, path) in enumerate(contact):
        image = Image.open(path).convert("RGB")
        image.thumbnail((780, 390), Image.Resampling.LANCZOS)
        x = (i % 2) * 800 + 10
        y = (i // 2) * 450 + 48
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x + 6, y - 38), title, font=font(25), fill="#f4f8ff")
    sheet.save(OUTPUT / "8个场景配准总览.jpg", quality=92)

    readme = f"""# 人工标注全景影像—原始 LiDAR 配准场景包

本包为 v3 指定树种版，共 8 个场景：尖沙咀 5 个、司徒拔道 3 个。06、07 保留上版狐尾椰子帧；其余六个替换为红花羊蹄甲、榕树、石栗相关场景。每帧均有至少 2 个有效人工标注。计数属于跨帧标注实例，不代表去重后的实体树数。原标签中的细叶榕、垂叶榕保留原名，汇总时统称榕树。

旧版的尖沙咀坐标变换误用了普通平面角度；v2 直接复用原项目 INS 航向函数，北向为零、顺时针，旋转顺序 level @ roll @ pitch。旧版尖沙咀 PCD 和配准图片不应继续使用。

## 数据约束

- 每个场景最多输出 150000 点，不足时保留所有候选点。WHU 的 8192 点是单树分类输入规模，不适合作为整条街景的上限。
- 未使用 CEDD 掩膜、SegmentAnyTree、地面去除、去噪、聚类或语义筛除。
- 只执行时空裁剪、坐标变换和确定性降采样。
- panorama_original.jpg 为原始 JPG 的逐字节复制；人工标注图缩放到 2048×1024，未套用曝光/对比度调整。图中树号对应 annotation.json 中从 1 开始的标签顺序。
- 尖沙咀及司徒拔道采用源建图点云，按相机位置 45 m 水平半径空间匹配；使用厂商 DS 文件构建的 0.15 m 体素缓存，再根据 150000 点上限降采样。建图文件无逐点时间，不能据此声称已验证逐影像 ±3 秒的时间窗口。
- 精细外参未提供，配准图使用原项目坐标轴和 180° 偏航关系，因此是可视化核验图，不应写成标定真值。修正航向计算不等同于完成精密外参标定。
- PCD 使用相机局部坐标，单位米，X前/Y左/Z上；全景公式 u=0.5-atan2(y,x)/(2π), v=0.5-asin(z/r)/π。因是多扫描/建图点云，视角遮挡差异和运动目标重影仍可能存在。

## 文件结构

每个场景包含：`点云数据/lidar_scene_sparse.pcd`、`影像数据/panorama_original.jpg`、`影像数据/panorama_annotated.jpg`、`影像数据/annotation.json`、`配准图片/registration_overlay.jpg`、`场景说明.json`。

详细来源、匹配方法、相机姿态、采样随机种子和生成文件 SHA-256 均在每个 `场景说明.json` 中。总表见 `总清单.csv`。
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"DONE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    build()
