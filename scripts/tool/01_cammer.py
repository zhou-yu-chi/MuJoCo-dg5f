import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def save_uint16_png(path: Path, depth_image: np.ndarray):
    if depth_image.dtype != np.uint16:
        raise ValueError("depth_image 必須是 uint16")
    ok = cv2.imwrite(str(path), depth_image)
    if not ok:
        raise RuntimeError(f"無法寫入深度圖：{path}")


def depth_to_vis(depth_image: np.ndarray):
    vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis


def make_pointcloud_from_arrays(color_image, depth_image, intrinsics, depth_scale, depth_trunc=1.5):
    """
    回傳的點雲是標準相機座標系（X 右、Y 下、Z 朝場景內，跟 DexGraspNet2
    的 depth_image_to_point_cloud() 完全一致）。這份是要落地存檔、餵給
    DexGraspNet2 pipeline 用的，不要在這裡做任何視覺化用的翻轉。
    """
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0 / depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.ppx,
        intrinsics.ppy
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)

    return pcd


def flip_for_display(pcd: o3d.geometry.PointCloud):
    """
    只給 o3d.visualization 預覽窗口用的翻轉版本（Y、Z 取負，符合 Open3D
    檢視器的 OpenGL 相機習慣，單純讓畫面看起來是正的）。
    不要把這個版本存檔或送進 DexGraspNet2，否則座標系會跟訓練資料不一致。
    """
    flip = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    display_pcd = o3d.geometry.PointCloud(pcd)
    display_pcd.transform(flip)
    return display_pcd


def ensure_not_empty_pcd(pcd, name="點雲"):
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"{name} 是空的，請重新拍攝或重框")


def center_pointcloud(pcd: o3d.geometry.PointCloud):
    ensure_not_empty_pcd(pcd, "center 前點雲")
    centered = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(centered.points)
    center = pts.mean(axis=0)
    pts = pts - center
    centered.points = o3d.utility.Vector3dVector(pts)

    if pcd.has_colors():
        centered.colors = pcd.colors

    return centered, center.tolist()


def clean_pointcloud(pcd: o3d.geometry.PointCloud, nb_neighbors=20, std_ratio=2.0):
    if len(pcd.points) == 0:
        return pcd

    cleaned = o3d.geometry.PointCloud(pcd)

    if len(cleaned.points) > 50:
        cleaned, _ = cleaned.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

    if len(cleaned.points) > 50:
        cleaned, _ = cleaned.remove_radius_outlier(
            nb_points=12,
            radius=0.02
        )

    return cleaned


def create_realsense_post_filters(
    spatial_magnitude=2,
    spatial_alpha=0.5,
    spatial_delta=20,
    spatial_holes_fill=2,
    temporal_alpha=0.4,
    temporal_delta=20,
    temporal_persistency=3,
    hole_fill_mode=1,
):
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, spatial_magnitude)
    spatial.set_option(rs.option.filter_smooth_alpha, spatial_alpha)
    spatial.set_option(rs.option.filter_smooth_delta, spatial_delta)
    try:
        spatial.set_option(rs.option.holes_fill, spatial_holes_fill)
    except Exception:
        pass

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, temporal_alpha)
    temporal.set_option(rs.option.filter_smooth_delta, temporal_delta)
    try:
        temporal.set_option(rs.option.holes_fill, temporal_persistency)
    except Exception:
        pass

    try:
        hole_filling = rs.hole_filling_filter(hole_fill_mode)
    except TypeError:
        hole_filling = rs.hole_filling_filter()

    return {
        "depth_to_disparity": depth_to_disparity,
        "disparity_to_depth": disparity_to_depth,
        "spatial": spatial,
        "temporal": temporal,
        "hole_filling": hole_filling,
    }


def apply_realsense_post_filters(depth_frame, filters):
    filtered = depth_frame
    filtered = filters["depth_to_disparity"].process(filtered)
    filtered = filters["spatial"].process(filtered)
    filtered = filters["temporal"].process(filtered)
    filtered = filters["disparity_to_depth"].process(filtered)
    filtered = filters["hole_filling"].process(filtered)
    return filtered


def _safe_percentile(values, q, fallback=0.0):
    if values is None or len(values) == 0:
        return fallback
    return float(np.percentile(values, q))


def _score_contour(
    cnt,
    roi_shape,
    weight_area=4.0,
    weight_center=2.0,
    weight_solidity=2.0,
    weight_extent=0.75,
    border_penalty=0.75,
):
    h, w = roi_shape[:2]
    roi_area = float(h * w)

    area = float(cv2.contourArea(cnt))
    if area <= 1.0:
        return -1e9, {}

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    x, y, bw, bh = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(hull_area, 1.0)
    extent = area / max(float(bw * bh), 1.0)

    center_dist = np.hypot(cx - w / 2.0, cy - h / 2.0)
    center_dist_norm = center_dist / max(np.hypot(w, h), 1.0)

    touches_border = (x <= 1) or (y <= 1) or ((x + bw) >= (w - 2)) or ((y + bh) >= (h - 2))

    score = (
        weight_area * (area / roi_area)
        + weight_solidity * solidity
        + weight_extent * extent
        - weight_center * center_dist_norm
        - (border_penalty if touches_border else 0.0)
    )

    info = {
        "area": area,
        "solidity": solidity,
        "extent": extent,
        "center": (float(cx), float(cy)),
        "touches_border": bool(touches_border),
    }
    return score, info


def _select_best_contour(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best_cnt = None
    best_score = -1e18
    best_info = None

    for cnt in contours:
        score, info = _score_contour(cnt, binary_mask.shape)
        if score > best_score:
            best_score = score
            best_cnt = cnt
            best_info = info

    return best_cnt, best_info


def _fill_holes_limited(depth_roi_u16, support_mask_u8, inpaint_radius=3):
    hole_mask = ((depth_roi_u16 == 0) & (support_mask_u8 > 0)).astype(np.uint8) * 255
    if not np.any(hole_mask):
        return depth_roi_u16.copy()

    depth_f32 = depth_roi_u16.astype(np.float32)
    filled_f32 = cv2.inpaint(depth_f32, hole_mask, float(inpaint_radius), cv2.INPAINT_TELEA)
    filled_f32 = np.clip(filled_f32, 0, np.iinfo(np.uint16).max)
    return filled_f32.astype(np.uint16)


def _build_grabcut_mask_from_depth(seed_mask_u8, depth_valid_u8):
    h, w = seed_mask_u8.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(
        seed_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    sure_fg = cv2.erode(
        seed_mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    sure_fg = cv2.bitwise_and(sure_fg, depth_valid_u8)

    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    border = max(3, int(round(min(h, w) * 0.02)))
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    probable_bg = cv2.dilate(
        probable_fg,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    probable_bg = cv2.bitwise_not(probable_bg)
    gc_mask[(probable_bg > 0) & (depth_valid_u8 == 0)] = cv2.GC_PR_BGD

    return gc_mask


def _finalize_binary_mask(mask_u8, close_ksize=7, open_ksize=3, edge_dilate_ksize=3):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_dilate_ksize, edge_dilate_ksize))

    out = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, open_kernel, iterations=1)
    out = cv2.dilate(out, dilate_kernel, iterations=1)
    out = cv2.medianBlur(out, 5)
    out = (out > 127).astype(np.uint8) * 255
    return out


def filter_object_depth_in_roi(
    color_image,
    depth_image,
    roi,
    intrinsics,
    depth_scale,
    depth_trunc=1.5,
    band_mm=80,
    min_depth_mm=100,
    use_grabcut="auto",
    grabcut_iters=2,
    debug=False,
):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("ROI 無效")

    color_roi = color_image[y:y+h, x:x+w].copy()
    depth_roi_raw = depth_image[y:y+h, x:x+w].copy().astype(np.uint16)

    if color_roi.size == 0 or depth_roi_raw.size == 0:
        raise RuntimeError("ROI 裁切為空")

    roi_area = float(w * h)
    depth_valid_raw = (depth_roi_raw > min_depth_mm)
    valid_values = depth_roi_raw[depth_valid_raw]
    valid_ratio = float(valid_values.size) / max(roi_area, 1.0)

    if len(valid_values) == 0:
        raise RuntimeError("ROI 內沒有有效深度")

    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    norm_r = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
    central_mask = (norm_r < 0.28) & depth_valid_raw
    central_valid = depth_roi_raw[central_mask]
    seed_values = central_valid if len(central_valid) >= max(30, 0.01 * roi_area) else valid_values

    obj_depth = int(np.median(seed_values))

    q10 = _safe_percentile(seed_values, 10, obj_depth)
    q25 = _safe_percentile(seed_values, 25, obj_depth)
    q75 = _safe_percentile(seed_values, 75, obj_depth)
    q90 = _safe_percentile(seed_values, 90, obj_depth)
    iqr = max(q75 - q25, 1.0)
    spread = max(q90 - q10, iqr)

    adaptive_band = int(max(band_mm, 0.75 * spread, 90))
    if valid_ratio < 0.12:
        adaptive_band = max(adaptive_band, 150)
    if valid_ratio < 0.06:
        adaptive_band = max(adaptive_band, 220)
    adaptive_band = int(np.clip(adaptive_band, 90, 320))

    lower = int(max(min_depth_mm, obj_depth - adaptive_band))
    upper = int(obj_depth + adaptive_band)

    rough_mask = (
        ((depth_roi_raw >= lower) & (depth_roi_raw <= upper)) & depth_valid_raw
    ).astype(np.uint8) * 255

    rough_mask = cv2.morphologyEx(
        rough_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    rough_mask = cv2.dilate(
        rough_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    depth_roi_filled = _fill_holes_limited(depth_roi_raw, rough_mask, inpaint_radius=3)
    depth_roi_filled = cv2.medianBlur(depth_roi_filled, 5)

    depth_for_mask = cv2.bilateralFilter(
        depth_roi_filled.astype(np.float32),
        d=5,
        sigmaColor=25.0,
        sigmaSpace=5.0,
    )

    depth_mask = (
        (depth_for_mask >= float(lower))
        & (depth_for_mask <= float(upper))
        & (depth_roi_filled > min_depth_mm)
    ).astype(np.uint8) * 255

    depth_mask = cv2.morphologyEx(
        depth_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    depth_mask = cv2.morphologyEx(
        depth_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    best_cnt, best_info = _select_best_contour(depth_mask)
    if best_cnt is None:
        raise RuntimeError("無法在 ROI 中找到候選物體輪廓")

    base_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(base_mask, [best_cnt], -1, 255, thickness=-1)
    base_mask = _finalize_binary_mask(base_mask, close_ksize=7, open_ksize=3, edge_dilate_ksize=3)

    use_grabcut_now = False
    if use_grabcut is True:
        use_grabcut_now = True
    elif use_grabcut == "auto":
        if valid_ratio < 0.15:
            use_grabcut_now = True
        elif best_info is not None and best_info.get("solidity", 1.0) < 0.80:
            use_grabcut_now = True
        elif best_info is not None and best_info.get("touches_border", False):
            use_grabcut_now = True

    final_mask = base_mask.copy()

    if use_grabcut_now:
        color_roi_gc = cv2.bilateralFilter(color_roi, d=7, sigmaColor=50, sigmaSpace=7)
        depth_valid_u8 = (depth_roi_filled > min_depth_mm).astype(np.uint8) * 255

        gc_mask = _build_grabcut_mask_from_depth(base_mask, depth_valid_u8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                color_roi_gc,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                grabcut_iters,
                cv2.GC_INIT_WITH_MASK,
            )
            gc_fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            support = cv2.dilate(
                rough_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                iterations=1,
            )
            final_mask = cv2.bitwise_and(gc_fg, support)

            base_core = cv2.erode(
                base_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            final_mask = cv2.bitwise_or(final_mask, base_core)
            final_mask = _finalize_binary_mask(final_mask, close_ksize=7, open_ksize=3, edge_dilate_ksize=3)

            gc_cnt, _ = _select_best_contour(final_mask)
            if gc_cnt is not None:
                tmp = np.zeros_like(final_mask)
                cv2.drawContours(tmp, [gc_cnt], -1, 255, thickness=-1)
                final_mask = _finalize_binary_mask(tmp, close_ksize=7, open_ksize=3, edge_dilate_ksize=3)
            else:
                final_mask = base_mask.copy()

        except cv2.error:
            final_mask = base_mask.copy()

    if np.count_nonzero(final_mask) == 0:
        raise RuntimeError("背景去除後沒有剩下物件，請重框或調整拍攝距離")

    masked_color_roi = color_roi.copy()
    masked_depth_roi = depth_roi_filled.copy()

    masked_color_roi[final_mask == 0] = 0
    masked_depth_roi[final_mask == 0] = 0

    color_only = np.zeros_like(color_image, dtype=np.uint8)
    depth_only = np.zeros_like(depth_image, dtype=np.uint16)
    full_mask = np.zeros_like(depth_image, dtype=np.uint8)

    color_only[y:y+h, x:x+w] = masked_color_roi
    depth_only[y:y+h, x:x+w] = masked_depth_roi
    full_mask[y:y+h, x:x+w] = final_mask

    object_pcd = make_pointcloud_from_arrays(
        color_only,
        depth_only,
        intrinsics,
        depth_scale,
        depth_trunc=depth_trunc,
    )
    object_pcd = clean_pointcloud(object_pcd, nb_neighbors=20, std_ratio=1.8)

    result = {
        "object_pcd": object_pcd,
        "color_roi": color_roi,
        "depth_roi": depth_roi_raw,
        "masked_color_roi": masked_color_roi,
        "masked_depth_roi": masked_depth_roi,
        "full_mask": full_mask,
        "estimated_object_depth_mm": obj_depth,
    }

    if debug:
        result["debug"] = {
            "valid_ratio": valid_ratio,
            "adaptive_band_mm": adaptive_band,
            "rough_mask": rough_mask,
            "depth_mask": depth_mask,
            "base_mask": base_mask,
            "final_mask_roi": final_mask,
            "depth_roi_filled": depth_roi_filled,
            "best_contour_info": best_info,
            "use_grabcut": use_grabcut_now,
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="captures")
    parser.add_argument("--scene_prefix", type=str, default="my_scene")
    parser.add_argument("--object_prefix", type=str, default="my_object")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--object_band_mm", type=int, default=60)
    parser.add_argument("--debug_masks", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    rs_filters = create_realsense_post_filters()

    print("\n=== 相機已開啟 ===")
    print("操作方式：")
    print("  p : 拍攝")
    print("  q : 離開")
    print("拍攝後請框選物件。\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            depth_frame = apply_realsense_post_filters(depth_frame, rs_filters)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            stacked = np.hstack((color_image, depth_to_vis(depth_image)))
            cv2.imshow("RealSense | Left: Color | Right: Depth", stacked)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("p"):
                frozen_color = color_image.copy()
                frozen_depth = depth_image.copy()

                roi = cv2.selectROI(
                    "Select Object ROI",
                    frozen_color,
                    showCrosshair=True,
                    fromCenter=False
                )
                cv2.destroyWindow("Select Object ROI")

                x, y, w, h = roi
                if w <= 0 or h <= 0:
                    print("ROI 無效，取消這次拍攝。")
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                scene_id = f"{args.scene_prefix}_{ts}"
                object_id = f"{args.object_prefix}_{ts}"

                capture_root = output_root / f"capture_{ts}"
                scene_dir = capture_root / "scene"
                object_dir = capture_root / "object"
                scene_dir.mkdir(parents=True, exist_ok=True)
                object_dir.mkdir(parents=True, exist_ok=True)

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                scene_color_path = scene_dir / "color.png"
                scene_depth_path = scene_dir / "depth.png"
                cv2.imwrite(str(scene_color_path), frozen_color)
                save_uint16_png(scene_depth_path, frozen_depth)

                scene_pcd = make_pointcloud_from_arrays(
                    frozen_color, frozen_depth, intr, depth_scale, depth_trunc=args.depth_trunc
                )
                ensure_not_empty_pcd(scene_pcd, "scene_raw 點雲")
                scene_pcd = clean_pointcloud(scene_pcd, nb_neighbors=20, std_ratio=2.0)

                scene_raw_ply = scene_dir / "scene_raw.ply"
                o3d.io.write_point_cloud(str(scene_raw_ply), scene_pcd)

                intrinsics_json = {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "cx": intr.ppx,
                    "cy": intr.ppy,
                    "depth_scale": depth_scale,
                    "depth_trunc_m": args.depth_trunc
                }
                intrinsics_path = scene_dir / "intrinsics.json"
                intrinsics_path.write_text(
                    json.dumps(intrinsics_json, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                scene_meta = {
                    "scene_id": scene_id,
                    "capture_time": ts,
                    "roi_for_object_reference": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "files": {
                        "color_png": str(scene_color_path.resolve()),
                        "depth_png": str(scene_depth_path.resolve()),
                        "scene_raw_ply": str(scene_raw_ply.resolve()),
                        "intrinsics_json": str(intrinsics_path.resolve())
                    }
                }
                scene_meta_path = scene_dir / "scene_meta.json"
                scene_meta_path.write_text(
                    json.dumps(scene_meta, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                result = filter_object_depth_in_roi(
                    frozen_color,
                    frozen_depth,
                    roi,
                    intr,
                    depth_scale,
                    depth_trunc=args.depth_trunc,
                    band_mm=args.object_band_mm,
                    debug=args.debug_masks
                )

                color_roi_path = object_dir / "color_roi.png"
                depth_roi_path = object_dir / "depth_roi.png"
                object_mask_path = object_dir / "object_mask.png"

                cv2.imwrite(str(color_roi_path), result["color_roi"])
                save_uint16_png(depth_roi_path, result["depth_roi"])
                cv2.imwrite(str(object_mask_path), result["full_mask"])

                masked_preview_path = object_dir / "masked_color_roi.png"
                cv2.imwrite(str(masked_preview_path), result["masked_color_roi"])

                object_pcd = result["object_pcd"]
                ensure_not_empty_pcd(object_pcd, "object_raw 點雲")

                object_raw_ply = object_dir / "object_raw.ply"
                o3d.io.write_point_cloud(str(object_raw_ply), object_pcd)

                object_centered_pcd, center_xyz = center_pointcloud(object_pcd)
                object_centered_ply = object_dir / "object_centered.ply"
                o3d.io.write_point_cloud(str(object_centered_ply), object_centered_pcd)

                object_meta = {
                    "object_id": object_id,
                    "capture_time": ts,
                    "roi": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "estimated_object_depth_mm": int(result["estimated_object_depth_mm"]),
                    "object_center_xyz_before_centering": center_xyz,
                    "files": {
                        "color_roi_png": str(color_roi_path.resolve()),
                        "depth_roi_png": str(depth_roi_path.resolve()),
                        "object_mask_png": str(object_mask_path.resolve()),
                        "masked_color_roi_png": str(masked_preview_path.resolve()),
                        "object_raw_ply": str(object_raw_ply.resolve()),
                        "object_centered_ply": str(object_centered_ply.resolve())
                    }
                }
                object_meta_path = object_dir / "object_meta.json"
                object_meta_path.write_text(
                    json.dumps(object_meta, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                capture_manifest = {
                    "capture_root": str(capture_root.resolve()),
                    "scene_dir": str(scene_dir.resolve()),
                    "object_dir": str(object_dir.resolve()),
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "created_at": ts
                }
                manifest_path = capture_root / "capture_manifest.json"
                manifest_path.write_text(
                    json.dumps(capture_manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                print("\n=== 拍攝完成 ===")
                print(f"capture_root = {capture_root}")
                print(f"scene_id     = {scene_id}")
                print(f"object_id    = {object_id}")

                cv2.imshow("Object Mask Preview", result["full_mask"])
                cv2.imshow("Masked Object ROI Preview", result["masked_color_roi"])

                if args.debug_masks and "debug" in result:
                    dbg = result["debug"]

                    def to_u8_vis(img):
                        if img.dtype == np.uint16:
                            return cv2.convertScaleAbs(img, alpha=255.0 / max(np.max(img), 1))
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            mx = float(np.max(img)) if np.max(img) > 0 else 1.0
                            return np.clip(img / mx * 255.0, 0, 255).astype(np.uint8)
                        return img

                    if "rough_mask" in dbg:
                        cv2.imshow("DEBUG rough_mask", dbg["rough_mask"])
                    if "depth_mask" in dbg:
                        cv2.imshow("DEBUG depth_mask", dbg["depth_mask"])
                    if "base_mask" in dbg:
                        cv2.imshow("DEBUG base_mask", dbg["base_mask"])
                    if "final_mask_roi" in dbg:
                        cv2.imshow("DEBUG final_mask_roi", dbg["final_mask_roi"])
                    if "depth_roi_filled" in dbg:
                        cv2.imshow("DEBUG depth_roi_filled", to_u8_vis(dbg["depth_roi_filled"]))

                cv2.waitKey(300)

                print("開啟場景點雲預覽...")
                o3d.visualization.draw_geometries(
                    [flip_for_display(scene_pcd)],
                    window_name="Scene Point Cloud Preview"
                )

                print("開啟物件點雲預覽...")
                o3d.visualization.draw_geometries(
                    [flip_for_display(object_pcd)],
                    window_name="Object Point Cloud Preview"
                )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()