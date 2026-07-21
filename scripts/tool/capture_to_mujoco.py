#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_to_mujoco.py  (重寫版 / rewrite)
========================================
單台 RealSense + 轉盤 + fiducial（ChArUco 板，或退而求其次的單張 ArUco）
掃描桌上小物件，融合成封閉 mesh，輸出成 MuJoCo 吃得下的物件資產。

跟舊版最重要的差別（也是你「rebuild 跟按 b 結果不一樣」的根因修正）
--------------------------------------------------------------------
舊版有「兩條會分岔的路」：
  * 按 b（現拍現建）：融合的是「拍攝當下記在記憶體裡的姿態」。
  * --rebuild（離線重建）：**預設會拿存檔的畫面『重新偵測一次 marker 姿態』**
    （redetect_marker=True），而且重新偵測時 marker 邊長、程式版本只要有一點
    不同，算出來的姿態就不一樣 → 融合結果就不一樣。

本版把架構改成「只有一條路」：
  * capture 只負責把**原始資料**（原始 color、原始 depth、當下算好的姿態、
    內參、ROI、marker 邊長）落地存檔，**不做任何融合、不做任何破壞性處理**。
  * reconstruct 是一個「純函式」：吃存檔資料 → 去背 → 砍轉盤 → 侵蝕 → TSDF
    融合 → 清乾淨 → 補底 → VHACD → object.xml。
  * 「現拍現建」與「離線重建」呼叫的是**同一個 reconstruct()**，讀的是**同一批
    存檔資料**，姿態直接沿用存檔那一份、預設不再重算。
  => 只要輸入資料一樣，輸出必然一樣。rebuild 不可能再跟 b 分岔。

而且因為存的是「原始 depth」而不是「已經去背砍好的 depth」，所以連
--plate_z_clip_m / --mask_erode_px / --tsdf_* 這些參數都能在 reconstruct
階段離線重調，真正做到「拍一次、之後反覆重建」。

指令
----
  # 1) 第一次先印一張 fiducial 板（強烈建議用 ChArUco，比單張 marker 穩非常多）
  python capture_to_mujoco.py make-board --out board.png

  # 2) 開相機拍攝（把物件放轉盤中央、板貼在轉盤上跟物件一起轉、相機不動）
  python capture_to_mujoco.py capture --object_name my_cup

  # 3) 之後只想調重建參數、不想重拍：
  python capture_to_mujoco.py reconstruct --object_name my_cup --tsdf_voxel_length 0.002

操作鍵（capture 視窗內）
  p : 拍一個視角（框物體 ROI；轉一格再拍，繞一圈 12~20 張）
  b : 用目前所有視角「離線重建」+ 開 MuJoCo 預覽（跟 reconstruct 指令同一條路）
  c : 清空目前視角
  q : 離開

作者備註：本檔刻意做成「不依賴 01_cammer.py / 06 / 08」的獨立檔，方便單獨執行；
若要接回你原本的抓握 pipeline，把最後 build_mujoco_object() 產生的 object.xml
拿去 include 即可，格式與 YCB 物件一致。
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# 這幾個重的相依只有在真的要用到時才 import，讓 make-board / 語法檢查在沒裝
# open3d/mujoco 的機器上也能跑。
def _lazy_import_o3d():
    import open3d as o3d
    return o3d

def _lazy_import_trimesh():
    import trimesh
    return trimesh

def _lazy_import_rs():
    import pyrealsense2 as rs
    return rs


# ===========================================================================
# 0. 板 / fiducial 設定
# ===========================================================================
# 為什麼預設用 ChArUco 而不是單張 ArUco：
#   單張平面正方形 marker 估姿態有「翻轉歧義」——斜看時正確姿態跟它的鏡射解
#   在影像上投影幾乎一樣，很容易挑錯，一挑錯那一幀就整個被丟到錯的位置，融合
#   出來就是一根根放射狀尖刺 / 一坨糊掉的 blob。而且 marker 一小、一被物體或
#   手擋一下，整幀就作廢。
#   ChArUco = 棋盤格 + 一堆 ArUco，角點是「棋盤內角」有 subpixel 精度、又有幾
#   十個點一起解 PnP，姿態穩非常多、也不怕被遮住一部分。學術界做轉盤/多視角
#   物件資料集幾乎都是用 marker 板（ArUco board / ChArUco），不是單張 marker。

@dataclass
class BoardSpec:
    kind: str                 # "charuco" 或 "single"
    dict_name: str            # e.g. "DICT_4X4_50"
    # charuco 用：
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.030    # 棋盤一格的邊長（公尺）
    marker_length_m: float = 0.022    # 格子裡 ArUco 的邊長（公尺）
    # single 用：
    marker_id: int = 0
    single_marker_length_m: float = 0.037

    def dictionary(self):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))


def make_charuco_board(spec: BoardSpec):
    # OpenCV >= 4.7 的新 API
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        spec.dictionary(),
    )


def render_board_png(spec: BoardSpec, out_path: Path, dpi: int = 300):
    """把板畫成可列印的 PNG。列印時務必選「實際大小 / 100%」，不要縮放。"""
    if spec.kind == "charuco":
        board = make_charuco_board(spec)
        w_m = spec.squares_x * spec.square_length_m
        h_m = spec.squares_y * spec.square_length_m
        px_w = int(round(w_m / 0.0254 * dpi))
        px_h = int(round(h_m / 0.0254 * dpi))
        img = board.generateImage((px_w, px_h), marginSize=int(dpi * 0.1))
        note = (f"ChArUco {spec.squares_x}x{spec.squares_y} | "
                f"square={spec.square_length_m*1000:.1f}mm marker={spec.marker_length_m*1000:.1f}mm "
                f"| dict={spec.dict_name}")
    else:
        dictionary = spec.dictionary()
        px = int(round(spec.single_marker_length_m / 0.0254 * dpi))
        marker = cv2.aruco.generateImageMarker(dictionary, spec.marker_id, px)
        margin = px // 4
        img = np.full((px + 2 * margin, px + 2 * margin), 255, np.uint8)
        img[margin:margin + px, margin:margin + px] = marker
        note = (f"ArUco id={spec.marker_id} | length={spec.single_marker_length_m*1000:.1f}mm "
                f"| dict={spec.dict_name}")

    cv2.imwrite(str(out_path), img)
    print(f"已產生板：{out_path.resolve()}")
    print(f"  {note}")
    print("  列印設定選『實際大小 / 100%』（不要『縮放至頁面』）。")
    print("  印出來後用尺量『棋盤一格』(或單 marker 的黑框) 實際邊長，")
    print("  跟上面的數字不一樣就用對應參數改掉，量錯整個模型比例會跟著錯。")


# ===========================================================================
# 1. 姿態偵測：影像 -> (camera -> world) 4x4
# ===========================================================================
# world 座標系 = 板座標系。板平放在轉盤上、印刷面朝上時，板的 Z 軸剛好是現實
# 世界的「垂直向上」，所以直接拿板座標系當 MuJoCo 世界座標系用，Z 自動朝上。

def _camera_matrix(intr) -> np.ndarray:
    return np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0, 0, 1]], dtype=np.float64)


def detect_pose_charuco(gray, intr, spec: BoardSpec,
                        min_corners: int = 6) -> Optional[Tuple[np.ndarray, float]]:
    board = make_charuco_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < min_corners:
        return None

    cam_mtx = _camera_matrix(intr)
    dist = np.array(intr.coeffs[:5], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    # 重投影誤差（像素 RMS），拿來當品質分數
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvec.flatten()
    cam_to_world = np.linalg.inv(world_to_cam)
    return cam_to_world, err


_SINGLE_OBJ_UNIT = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64)

def detect_pose_single(gray, intr, spec: BoardSpec,
                       max_reproj_px: float = 3.0) -> Optional[Tuple[np.ndarray, float]]:
    dictionary = spec.dictionary()
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.flatten()
    if spec.marker_id not in ids:
        return None
    idx = int(np.where(ids == spec.marker_id)[0][0])

    cam_mtx = _camera_matrix(intr)
    dist = np.array(intr.coeffs[:5], dtype=np.float64)
    obj_pts = _SINGLE_OBJ_UNIT * (spec.single_marker_length_m / 2.0)
    # IPPE_SQUARE 專給方形 marker，一次回兩個候選＋各自誤差，挑誤差小的抗翻轉歧義
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj_pts, corners[idx], cam_mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or len(rvecs) == 0:
        return None
    best = int(np.argmin(errs))
    err = float(errs[best].item())
    if err > max_reproj_px:
        return None
    R, _ = cv2.Rodrigues(rvecs[best])
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvecs[best].flatten()
    return np.linalg.inv(world_to_cam), err


def detect_pose(color_bgr, intr, spec: BoardSpec) -> Optional[Tuple[np.ndarray, float]]:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    if spec.kind == "charuco":
        return detect_pose_charuco(gray, intr, spec)
    return detect_pose_single(gray, intr, spec)


# ===========================================================================
# 2. RealSense 包裝（自帶「拍黑色物體」該有的設定，不依賴 01_cammer.py）
# ===========================================================================
# 你這台拍的是「黑色 + 反光」的鏡頭/杯子，這是深度相機的天敵：黑色會吸收
# 紅外線、反光會鏡面反射，主動式紅外深度相機在這種表面上量到的深度又稀又
# 亂又有偏差——這是物理，不是程式 bug，也是你重建出來一坨糊的最大單一主因。
# 能做的補救（程式面 + 物理面）：
#   [程式] 雷射打到最強 + 用 High Accuracy/Density preset + 多幀 temporal 濾波
#   [物理] 打一盞強燈照物體、或噴一層薄的消光噴劑/爽身粉讓表面變霧面（效果
#          遠比任何程式設定大，做物件掃描的人幾乎都會這樣處理反光/深色件）

class RealSenseCamera:
    def __init__(self, width=848, height=480, fps=30, preset="high_accuracy"):
        rs = _lazy_import_rs()
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        dev = self.profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # --- 針對深色/反光件的深度設定 ---
        try:  # 深度 preset
            self._apply_preset(depth_sensor, preset)
        except Exception as e:
            print(f"[提醒] 設定 depth preset 失敗（沿用預設）：{e}")
        try:  # 打開並打強紅外雷射，讓黑色件多少反射一點回來
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, rng.max)
        except Exception as e:
            print(f"[提醒] 設定雷射功率失敗：{e}")

        # --- 後處理濾波鏈（Intel 建議順序）---
        self.dec = rs.decimation_filter()          # 降取樣，去雜訊、加速
        self.dec.set_option(rs.option.filter_magnitude, 2)
        self.d2d = rs.disparity_transform(True)     # depth -> disparity
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal = rs.temporal_filter()        # 多幀時間平均（靜物超有效）
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.d2depth = rs.disparity_transform(False)
        # 刻意「不」用 hole_filling：它會用鄰居硬補出假深度，單幀好看，多視角
        # 融合時那些假深度會疊成貼地板往外攤的薄殼/裙擺。寧可留洞讓別的角度補。

    def _apply_preset(self, depth_sensor, preset):
        rs = self.rs
        if not depth_sensor.supports(rs.option.visual_preset):
            return
        want = {"high_accuracy": "High Accuracy",
                "high_density": "High Density",
                "default": "Default"}.get(preset, "High Accuracy")
        rng = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(rng.min), int(rng.max) + 1):
            name = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            if name == want:
                depth_sensor.set_option(rs.option.visual_preset, i)
                return

    def grab(self, temporal_frames: int = 10):
        """
        拉一小串連續幀跑過 temporal 濾波再回傳最後一張——靜物拍攝時這招能把
        深度雜訊壓下去一大截，對黑色件尤其有感。回傳對齊後的 (color_bgr,
        depth_uint16, intrinsics)。
        """
        rs = self.rs
        color_bgr, depth_raw, intr = None, None, None
        for _ in range(max(1, temporal_frames)):
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                continue
            f = self.dec.process(d)
            f = self.d2d.process(f)
            f = self.spatial.process(f)
            f = self.temporal.process(f)   # 逐幀餵進去，內部會累積時間資訊
            f = self.d2depth.process(f)
            depth_raw = np.asanyarray(f.as_depth_frame().get_data()).copy()
            color_bgr = np.asanyarray(c.get_data()).copy()
            intr = c.profile.as_video_stream_profile().intrinsics
        if color_bgr is None:
            raise RuntimeError("拿不到影像幀")
        # decimation 會把深度尺寸縮小，這裡把深度放回 color 尺寸，維持像素對齊
        if depth_raw.shape[:2] != color_bgr.shape[:2]:
            depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return color_bgr, depth_raw, intr

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ===========================================================================
# 3. 一個視角的原始資料（單一真相來源）
# ===========================================================================
# 注意：這裡存的 depth 是「原始、只在 ROI 內、沒有去背沒有砍轉盤沒有侵蝕」的
# 深度。所有破壞性處理都留到 reconstruct 階段做，這樣那些參數才能離線重調。

@dataclass
class ViewRecord:
    index: int
    color_file: str
    depth_file: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: List[float]
    depth_scale: float
    roi: List[int]                 # [x, y, w, h]
    pose_cam_to_world: List[List[float]]
    reproj_err_px: float


class _Intr:
    """給 detect_pose 用的極簡內參物件（reconstruct 離線重算姿態時才需要）。"""
    def __init__(self, fx, fy, cx, cy, coeffs):
        self.fx, self.fy, self.ppx, self.ppy = fx, fy, cx, cy
        self.coeffs = list(coeffs)


def save_view(debug_dir: Path, index: int, color_bgr, depth_u16, intr,
              depth_scale, roi, pose, err) -> ViewRecord:
    debug_dir.mkdir(parents=True, exist_ok=True)
    color_name = f"view_{index:02d}_color.png"
    depth_name = f"view_{index:02d}_depth.png"
    cv2.imwrite(str(debug_dir / color_name), color_bgr)
    # 16-bit 單通道 PNG，無損存原始深度
    cv2.imwrite(str(debug_dir / depth_name), depth_u16.astype(np.uint16))
    rec = ViewRecord(
        index=index, color_file=color_name, depth_file=depth_name,
        width=int(intr.width) if hasattr(intr, "width") else color_bgr.shape[1],
        height=int(intr.height) if hasattr(intr, "height") else color_bgr.shape[0],
        fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy),
        coeffs=[float(c) for c in intr.coeffs[:5]],
        depth_scale=float(depth_scale), roi=[int(v) for v in roi],
        pose_cam_to_world=[[float(x) for x in row] for row in pose],
        reproj_err_px=float(err),
    )
    return rec


def write_manifest(debug_dir: Path, records: List[ViewRecord], spec: BoardSpec):
    manifest = {
        "board": asdict(spec),
        "views": [asdict(r) for r in records],
    }
    (debug_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(debug_dir: Path):
    p = debug_dir / "manifest.json"
    if not p.exists():
        raise RuntimeError(f"找不到 {p}，這個物件還沒拍過任何視角，無法離線重建。")
    m = json.loads(p.read_text(encoding="utf-8"))
    spec = BoardSpec(**m["board"])
    recs = [ViewRecord(**r) for r in m["views"]]
    return recs, spec


# ===========================================================================
# 4. reconstruct：唯一一條重建路徑（b 跟 reconstruct 都走這裡）
# ===========================================================================

def _flying_pixel_mask(depth_u16, depth_scale, thresh_m):
    """
    飛點過濾：深度相機在物體輪廓邊緣、以及反光/玻璃表面上，常吐出「深度跟四周
    鄰居差一大截」的孤立像素（floating pixels）。這些點反投影後會飄在物體前後
    方或旁邊，是 TSDF 融合長出放射狀刀片的主因。這裡把跟鄰居落差超過 thresh_m
    的像素標成壞點。
    """
    d = depth_u16.astype(np.float32) * depth_scale
    valid = depth_u16 > 0
    worst = np.zeros_like(d)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sd = np.roll(d, (dy, dx), (0, 1))
        sv = np.roll(valid, (dy, dx), (0, 1))
        diff = np.abs(d - sd)
        diff[~sv] = 0.0
        worst = np.maximum(worst, diff)
    return valid & (worst > thresh_m)


def _world_points(depth_u16, roi, intr_dict, depth_scale, pose, depth_trunc):
    """把 ROI 內、depth_trunc 內的有效像素反投影到世界座標。回傳 (ys, xs, world_xyz)。"""
    x, y, w, h = roi
    H, W = depth_u16.shape[:2]
    full = np.zeros((H, W), np.uint16)
    full[y:y + h, x:x + w] = depth_u16[y:y + h, x:x + w]
    z_m = full.astype(np.float32) * depth_scale
    full[(z_m <= 0) | (z_m > depth_trunc)] = 0
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return ys, xs, np.empty((0, 3))
    fx, fy, cx, cy = intr_dict["fx"], intr_dict["fy"], intr_dict["cx"], intr_dict["cy"]
    zc = full[ys, xs].astype(np.float64) * depth_scale
    xc = (xs - cx) / fx * zc
    yc = (ys - cy) / fy * zc
    world = (np.stack([xc, yc, zc, np.ones_like(zc)], 1) @ np.asarray(pose).T)[:, :3]
    return ys, xs, world


def _reproject_and_clip(depth_u16, roi, intr_dict, depth_scale, pose,
                        depth_trunc, plate_z_clip_m, mask_erode_px,
                        center_xy, max_radius_m, max_height_m, flying_thresh_m):
    """
    回傳「只留物體本體」的全畫面 uint16 深度。處理順序：
    ROI 框 → depth 截斷 → 飛點過濾 → 邊緣侵蝕 → 反投影到世界後做包圍盒裁切
    （砍掉轉盤面以下、比物體高太多、以及橫向離物體中心太遠的雜點）。
    """
    x, y, w, h = roi
    H, W = depth_u16.shape[:2]
    full = np.zeros((H, W), np.uint16)
    full[y:y + h, x:x + w] = depth_u16[y:y + h, x:x + w]

    z_m = full.astype(np.float32) * depth_scale
    full[(z_m <= 0) | (z_m > depth_trunc)] = 0

    if flying_thresh_m > 0:
        full[_flying_pixel_mask(full, depth_scale, flying_thresh_m)] = 0

    if mask_erode_px > 0:
        mask = (full > 0).astype(np.uint8)
        k = 2 * mask_erode_px + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
        full[mask == 0] = 0

    # 反投影到世界座標，做物體包圍盒裁切
    fx, fy, cx, cy = intr_dict["fx"], intr_dict["fy"], intr_dict["cx"], intr_dict["cy"]
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return full
    zc = full[ys, xs].astype(np.float64) * depth_scale
    xw = (xs - cx) / fx * zc
    yw = (ys - cy) / fy * zc
    world = (np.stack([xw, yw, zc, np.ones_like(zc)], 1) @ np.asarray(pose).T)[:, :3]

    drop = world[:, 2] < plate_z_clip_m                    # 轉盤面以下
    if max_height_m and max_height_m > 0:
        drop |= world[:, 2] > (plate_z_clip_m + max_height_m)   # 比物體高太多
    if max_radius_m and max_radius_m > 0:
        r = np.hypot(world[:, 0] - center_xy[0], world[:, 1] - center_xy[1])
        drop |= r > max_radius_m                            # 橫向離物體太遠
    full[ys[drop], xs[drop]] = 0
    return full


def reconstruct(debug_dir: Path, args, spec: BoardSpec, records: List[ViewRecord]):
    """
    純函式：吃存好的視角資料，吐出 (open3d mesh, 世界座標)。b 與 reconstruct
    指令都呼叫這裡，所以只要 records 一樣、參數一樣，結果一定一樣。
    """
    o3d = _lazy_import_o3d()

    if len(records) < args.min_views:
        raise RuntimeError(f"只有 {len(records)} 個視角，至少需要 {args.min_views} 個。")

    # (可選) 離線重新偵測姿態：預設關閉。這正是舊版讓 rebuild 跟 b 分岔的元凶，
    # 這裡預設沿用存檔姿態，只有你明確 --redetect 才會重算。
    if args.redetect:
        print("[--redetect] 用存檔畫面重新偵測姿態（會覆蓋存檔姿態）...")
        for r in records:
            color = cv2.imread(str(debug_dir / r.color_file))
            intr = _Intr(r.fx, r.fy, r.cx, r.cy, r.coeffs)
            got = detect_pose(color, intr, spec)
            if got is not None:
                r.pose_cam_to_world = [[float(x) for x in row] for row in got[0]]
                r.reproj_err_px = got[1]

    # 姿態離群值剔除（單 marker 翻轉歧義的保險，charuco 幾乎用不到但留著無害）
    kept = reject_pose_outliers(records, args.max_neighbor_pose_jump_deg)
    if len(kept) < 3:
        raise RuntimeError("剔除異常姿態後剩不到 3 個視角，請重拍並確保板全程清楚可見。")

    # --- 前置 pass：估物體水平中心 ---
    # 把所有視角在「轉盤面以上」的點聚起來，用 2D 直方圖找最密的格子當中心。
    # 用「最密」而不是平均/中位數，是因為刀片雜點雖然飄很遠但數量少，峰值一定
    # 落在物體本體上，對雜點免疫。之後就以這個中心裁掉橫向太遠的點。
    center_xy = (0.0, 0.0)
    if args.object_max_radius_m and args.object_max_radius_m > 0:
        acc = []
        per_view_xy = []   # (index, cx_i, cy_i, n_pts)，之後拿來做打滑偵測
        for r in kept:
            depth_u16 = cv2.imread(str(debug_dir / r.depth_file), cv2.IMREAD_UNCHANGED)
            intr_dict = {"fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy}
            _, _, world = _world_points(depth_u16, r.roi, intr_dict, r.depth_scale,
                                        np.asarray(r.pose_cam_to_world), args.depth_trunc)
            if len(world):
                above = world[world[:, 2] > args.plate_z_clip_m]
                if len(above):
                    acc.append(above[:, :2])
                    if len(above) >= 50:
                        hxi, exi = np.histogram(above[:, 0], bins=30)
                        hyi, eyi = np.histogram(above[:, 1], bins=30)
                        cxi = (exi[hxi.argmax()] + exi[hxi.argmax() + 1]) / 2
                        cyi = (eyi[hyi.argmax()] + eyi[hyi.argmax() + 1]) / 2
                        per_view_xy.append((r.index, float(cxi), float(cyi), len(above)))
        if acc:
            xy = np.vstack(acc)
            hx, ex = np.histogram(xy[:, 0], bins=60)
            hy, ey = np.histogram(xy[:, 1], bins=60)
            cx0 = (ex[hx.argmax()] + ex[hx.argmax() + 1]) / 2
            cy0 = (ey[hy.argmax()] + ey[hy.argmax() + 1]) / 2
            center_xy = (float(cx0), float(cy0))
        print(f"物體水平中心估計：x={center_xy[0]*100:.1f}cm y={center_xy[1]*100:.1f}cm"
              f"（裁切半徑 {args.object_max_radius_m*100:.0f}cm、高度上限 {args.object_max_height_m*100:.0f}cm）")

        # 打滑偵測：理論上物體跟板一起轉，world 座標系裡物體中心應該固定不動。
        # 如果某個視角的物體重心離整體估計的中心太遠，代表拍攝過程中物體在轉盤
        # 上滑動/移位了——這正是融合出「放射狀尖刺花朵」的頭號原因，遠比深度
        # 雜訊常見，值得在建 mesh 之前就先抓出來。
        drift_thresh = args.max_center_drift_m
        drifted = [(idx, cxi, cyi, n, float(np.hypot(cxi - center_xy[0], cyi - center_xy[1])))
                   for idx, cxi, cyi, n in per_view_xy]
        bad = [d for d in drifted if d[4] > drift_thresh]
        if bad:
            print(f"[警告] 偵測到疑似物體打滑：{len(bad)}/{len(drifted)} 個視角的重心"
                  f"偏離整體中心超過 {drift_thresh*100:.1f}cm（可能是物體在轉盤上沒固定牢，"
                  f"拍攝時滑動了）：")
            for idx, cxi, cyi, n, dd in sorted(bad, key=lambda d: -d[4]):
                print(f"    視角 {idx:2d}: 偏移 {dd*100:5.1f}cm（重心=({cxi*100:.1f},{cyi*100:.1f})cm, {n}點）")
            print("  建議：用雙面膠/黏土/止滑墊把物體牢牢固定在轉盤上再重拍，"
                  "單靠調參數救不回「物體本身在動」的資料。")

    # TSDF 融合
    print(f"TSDF 融合中（{len(kept)} 視角，voxel={args.tsdf_voxel_length}m）...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.tsdf_voxel_length,
        sdf_trunc=args.tsdf_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for r in kept:
        color = cv2.imread(str(debug_dir / r.color_file))
        depth_u16 = cv2.imread(str(debug_dir / r.depth_file), cv2.IMREAD_UNCHANGED)
        intr_dict = {"fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy}
        depth_clean = _reproject_and_clip(
            depth_u16, r.roi, intr_dict, r.depth_scale,
            np.asarray(r.pose_cam_to_world),
            args.depth_trunc, args.plate_z_clip_m, args.mask_erode_px,
            center_xy, args.object_max_radius_m, args.object_max_height_m,
            args.flying_px_thresh_m)

        color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_clean)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0 / r.depth_scale, depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            r.width, r.height, r.fx, r.fy, r.cx, r.cy)
        extrinsic = np.linalg.inv(np.asarray(r.pose_cam_to_world))  # world->cam
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF 融合後沒有三角形，多半是每一幀深度都被砍光了"
                           "（黑色物體深度太少 / plate_z_clip 太大 / ROI 沒框到物體）。")

    # mesh 層再裁一次包圍盒（保險）：即使有雜點漏進 TSDF，這裡也會被切掉
    if args.object_max_radius_m and args.object_max_radius_m > 0:
        v = np.asarray(mesh.vertices)
        if len(v):
            r = np.hypot(v[:, 0] - center_xy[0], v[:, 1] - center_xy[1])
            keep = (r <= args.object_max_radius_m) & (v[:, 2] >= args.plate_z_clip_m - 0.005)
            if args.object_max_height_m and args.object_max_height_m > 0:
                keep &= v[:, 2] <= (args.plate_z_clip_m + args.object_max_height_m)
            mesh.remove_vertices_by_mask(~keep)

    # 只留最大連通塊，丟掉背景漏進來的碎片
    if len(mesh.triangles) == 0:
        raise RuntimeError("包圍盒裁切後沒有三角形了：--object_max_radius_m / "
                           "--object_max_height_m 可能設太小，或物體沒放在板中央。")
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    biggest = int(counts.argmax())
    mesh.remove_triangles_by_mask(labels != biggest)
    mesh.remove_unreferenced_vertices()

    # 存一份世界座標的融合結果供除錯
    o3d.io.write_triangle_mesh(str(debug_dir / "fused_world.ply"), mesh)
    return mesh


def _rot_angle_deg(Ra, Rb):
    R = np.asarray(Ra).T @ np.asarray(Rb)
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def reject_pose_outliers(records: List[ViewRecord], max_jump_deg: float):
    if len(records) < 3:
        return list(records)
    keep = [True] * len(records)
    for i in range(1, len(records) - 1):
        Rp = np.asarray(records[i - 1].pose_cam_to_world)[:3, :3]
        Rc = np.asarray(records[i].pose_cam_to_world)[:3, :3]
        Rn = np.asarray(records[i + 1].pose_cam_to_world)[:3, :3]
        if (_rot_angle_deg(Rp, Rc) > max_jump_deg and
                _rot_angle_deg(Rc, Rn) > max_jump_deg and
                _rot_angle_deg(Rp, Rn) < max_jump_deg):
            keep[i] = False
    dropped = [r.index for r, k in zip(records, keep) if not k]
    if dropped:
        print(f"[警告] 視角 {dropped} 姿態跟前後鄰居差太多（疑似翻轉歧義），已排除。")
    return [r for r, k in zip(records, keep) if k]


# ===========================================================================
# 5. mesh 後處理 + MuJoCo 物件輸出
# ===========================================================================

def cap_bottom_hole(tri_mesh):
    """物件貼轉盤那面永遠沒被拍到，會留一個洞。找邊界迴圈扇形補一塊平面蓋住。"""
    edges = tri_mesh.edges_sorted
    view = edges.view([("", edges.dtype)] * 2).ravel()
    _, inv, cnt = np.unique(view, return_inverse=True, return_counts=True)
    boundary = edges[cnt[inv] == 1]
    if len(boundary) == 0:
        return tri_mesh
    adj = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    trimesh = _lazy_import_trimesh()
    verts = list(tri_mesh.vertices)
    faces = [tuple(f) for f in tri_mesh.faces]
    visited = set()
    for a0, b0 in boundary:
        a0, b0 = int(a0), int(b0)
        key = (min(a0, b0), max(a0, b0))
        if key in visited:
            continue
        loop = [a0, b0]; visited.add(key); closed = False
        for _ in range(len(boundary) + 2):
            cur, prev = loop[-1], loop[-2]
            nbrs = [n for n in adj.get(cur, []) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == a0:
                closed = True; break
            ek = (min(cur, nxt), max(cur, nxt))
            if ek in visited:
                break
            visited.add(ek); loop.append(nxt)
        if not closed or len(loop) < 3:
            continue
        c = np.mean([tri_mesh.vertices[i] for i in loop], axis=0)
        ci = len(verts); verts.append(c)
        for i in range(len(loop)):
            faces.append((loop[i], loop[(i + 1) % len(loop)], ci))
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)


def make_collision(tri_mesh, col_dir: Path, name: str, max_hulls: int):
    """
    凸分解成 collision hull。優先用 coacd（pip 裝得到、品質好），沒有就退到
    trimesh 內建 vhacd，再沒有就退到單一凸包（至少不會穿模，只是抓握手感差些）。
    回傳寫出的檔名清單。
    """
    trimesh = _lazy_import_trimesh()
    col_dir.mkdir(parents=True, exist_ok=True)
    files = []
    # 1) coacd
    try:
        import coacd
        coacd.set_log_level("error")
        m = coacd.Mesh(tri_mesh.vertices, tri_mesh.faces)
        parts = coacd.run_coacd(m, max_convex_hull=max_hulls)
        for i, (v, f) in enumerate(parts):
            fn = f"{name}_col_{i:02d}.stl"
            trimesh.Trimesh(vertices=v, faces=f, process=True).export(col_dir / fn)
            files.append(fn)
        if files:
            return files
    except Exception as e:
        print(f"[提醒] coacd 不可用（{e}），改用 trimesh vhacd。")
    # 2) trimesh vhacd：不同版本 / 不同 vhacd binding 的參數名不一樣
    #    （你遇到的錯是新版 binding 要 maxConvexHulls、不吃 maxhulls），這裡依序試
    last_e = "分解結果為空"
    for kwargs in ({"maxConvexHulls": max_hulls}, {"max_convex_hulls": max_hulls}, {}):
        try:
            hulls = tri_mesh.convex_decomposition(**kwargs)
            hulls = hulls if isinstance(hulls, list) else [hulls]
            hulls = [trimesh.Trimesh(**h) if isinstance(h, dict) else h for h in hulls]
            if hulls:
                for i, h in enumerate(hulls):
                    fn = f"{name}_col_{i:02d}.stl"
                    h.export(col_dir / fn); files.append(fn)
                print(f"  vhacd 分解成 {len(files)} 塊 collision hull。")
                return files
        except Exception as e:
            last_e = e
    print(f"[提醒] vhacd 不可用（{last_e}），退到單一凸包。"
          f"（建議 `pip install coacd` 取得可抓握的凸分解）")
    # 3) 單一凸包
    fn = f"{name}_col_00.stl"
    tri_mesh.convex_hull.export(col_dir / fn)
    return [fn]


VISUAL_ATTRS = 'contype="0" conaffinity="0" group="2" rgba="0.8 0.8 0.75 1"'
COLLISION_ATTRS = ('group="3" condim="4" friction="0.8 0.01 0.001" density="1000" '
                   'solref="0.005 1" solimp="0.9 0.98 0.001"')


def build_mujoco_object(mesh_o3d, obj_dir: Path, name: str, args):
    trimesh = _lazy_import_trimesh()
    V = np.asarray(mesh_o3d.vertices)
    F = np.asarray(mesh_o3d.triangles)
    tri = trimesh.Trimesh(vertices=V, faces=F, process=True)

    # 面數過多先簡化
    if args.max_visual_triangles and len(tri.faces) > args.max_visual_triangles:
        mesh_o3d2 = mesh_o3d.simplify_quadric_decimation(args.max_visual_triangles)
        tri = trimesh.Trimesh(vertices=np.asarray(mesh_o3d2.vertices),
                              faces=np.asarray(mesh_o3d2.triangles), process=True)

    if not tri.is_watertight:
        print("補底洞中（貼轉盤那面補一塊平面）...")
        tri = cap_bottom_hole(tri)

    # 重心置中（跟 YCB 物件一致，freejoint 物件才不會一放就亂飛）
    com = tri.center_mass if tri.is_watertight else tri.centroid
    tri.apply_translation(-com)

    obj_dir.mkdir(parents=True, exist_ok=True)
    visual_path = obj_dir / "visual.stl"
    tri.export(visual_path)

    col_dir = obj_dir / "collision"
    hull_files = make_collision(tri, col_dir, name, args.max_hulls)

    mesh_tags = [f'    <mesh name="{name}_visual" file="objects/{obj_dir.name}/visual.stl" />']
    geom_tags = [f'      <geom type="mesh" mesh="{name}_visual" {VISUAL_ATTRS} />']
    for i, fn in enumerate(hull_files):
        mn = f"{name}_col_{i:02d}"
        mesh_tags.append(f'    <mesh name="{mn}" file="objects/{obj_dir.name}/collision/{fn}" />')
        geom_tags.append(f'      <geom type="mesh" mesh="{mn}" {COLLISION_ATTRS} />')

    xml = ("<mujoco>\n  <asset>\n" + "\n".join(mesh_tags) + "\n  </asset>\n"
           "  <worldbody>\n"
           f'    <body name="{name}" pos="0 0 0.3">\n'
           f'      <freejoint name="{name}_free" />\n'
           + "\n".join(geom_tags) + "\n"
           "    </body>\n  </worldbody>\n</mujoco>\n")
    (obj_dir / "object.xml").write_text(xml)

    info = {"watertight": bool(tri.is_watertight),
            "triangles": int(len(tri.faces)),
            "collision_hulls": len(hull_files)}
    print(f"  watertight={info['watertight']} 三角形={info['triangles']} "
          f"collision hull={info['collision_hulls']}")
    return info


def write_preview_scene(assets_dir: Path, name: str) -> Path:
    scene = assets_dir / f"_preview_{name}.xml"
    scene.write_text(
        f'<mujoco model="{name}_preview">\n'
        '  <worldbody>\n'
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />\n'
        '    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 0" '
        'rgba="0.4 0.42 0.45 1" condim="3" friction="1.0 0.005 0.0001" />\n'
        '  </worldbody>\n'
        f'  <include file="objects/{name}/object.xml" />\n'
        '</mujoco>\n')
    return scene


def preview_in_mujoco(scene_path: Path):
    import mujoco
    import mujoco.viewer
    print(f"開 MuJoCo 預覽：{scene_path}（物件會從半空落下，關視窗回到相機）")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.002)


# ===========================================================================
# 6. 可微調的 ROI 選取（沿用舊版體驗：先拉大概再抓邊/角微調）
# ===========================================================================

def select_roi(window_name: str, image: np.ndarray) -> Tuple[int, int, int, int]:
    handle = 10
    st = {"box": None, "drag": None, "last": None}

    def sb(b):
        xa, ya, xb, yb = b
        return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)

    def hit(x, y, box):
        x0, y0, x1, y1 = sb(box)
        nl, nr = abs(x - x0) <= handle, abs(x - x1) <= handle
        nt, nb = abs(y - y0) <= handle, abs(y - y1) <= handle
        ix = x0 - handle <= x <= x1 + handle
        iy = y0 - handle <= y <= y1 + handle
        if nl and nt: return "topleft"
        if nr and nt: return "topright"
        if nl and nb: return "bottomleft"
        if nr and nb: return "bottomright"
        if nl and iy: return "left"
        if nr and iy: return "right"
        if nt and ix: return "top"
        if nb and ix: return "bottom"
        if x0 <= x <= x1 and y0 <= y <= y1: return "move"
        return None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if st["box"] is None:
                st["box"] = [x, y, x, y]; st["drag"] = "bottomright"
            else:
                h = hit(x, y, st["box"])
                if h is None:
                    st["box"] = [x, y, x, y]; h = "bottomright"
                st["drag"] = h
            st["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
            xa, ya, xb, yb = st["box"]; d = st["drag"]
            if d == "move":
                lx, ly = st["last"]; dx, dy = x - lx, y - ly
                st["box"] = [xa + dx, ya + dy, xb + dx, yb + dy]; st["last"] = (x, y); return
            if "left" in d: xa = x
            if "right" in d: xb = x
            if "top" in d: ya = y
            if "bottom" in d: yb = y
            st["box"] = [xa, ya, xb, yb]
        elif event == cv2.EVENT_LBUTTONUP:
            st["drag"] = None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        disp = image.copy()
        if st["box"]:
            x0, y0, x1, y1 = (int(v) for v in sb(st["box"]))
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(disp, "drag box/edges  ENTER=ok  c/ESC=cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32): break
        if k in (27, ord("c")): st["box"] = None; break
    cv2.destroyWindow(window_name)
    if st["box"] is None:
        return (0, 0, 0, 0)
    x0, y0, x1, y1 = sb(st["box"])
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# ===========================================================================
# 7. 指令
# ===========================================================================

def spec_from_args(args) -> BoardSpec:
    if args.board == "charuco":
        return BoardSpec(kind="charuco", dict_name=args.marker_dict,
                         squares_x=args.squares_x, squares_y=args.squares_y,
                         square_length_m=args.square_length_m,
                         marker_length_m=args.marker_length_m_charuco)
    return BoardSpec(kind="single", dict_name=args.marker_dict,
                     marker_id=args.marker_id,
                     single_marker_length_m=args.marker_length_m)


def cmd_make_board(args):
    render_board_png(spec_from_args(args), args.out)


def cmd_reconstruct(args, records=None, spec=None, live_camera_scene=None):
    assets = args.assets.resolve()
    name = args.object_name.replace("-", "_")
    obj_dir = assets / "objects" / name
    debug_dir = obj_dir / "capture_debug"
    if records is None:
        records, spec = read_manifest(debug_dir)
        print(f"讀回 {len(records)} 個視角，開始重建...")

    mesh = reconstruct(debug_dir, args, spec, records)
    build_mujoco_object(mesh, obj_dir, name, args)
    scene = write_preview_scene(assets, name)
    print(f"\n=== 完成 ===\nobject.xml = {obj_dir / 'object.xml'}\n預覽場景  = {scene}\n")
    if not args.no_view:
        preview_in_mujoco(scene)


def cmd_capture(args):
    assets = args.assets.resolve()
    name = args.object_name.replace("-", "_")
    obj_dir = assets / "objects" / name
    debug_dir = obj_dir / "capture_debug"
    spec = spec_from_args(args)

    cam = RealSenseCamera(args.width, args.height, args.fps, args.preset)
    records: List[ViewRecord] = []
    print("\n=== 相機已開啟 ===")
    print(f"物件：{name} -> {obj_dir}")
    print(f"板：{spec.kind}  dict={spec.dict_name}")
    print("p=拍一張  b=重建+預覽  c=清空  q=離開\n")
    try:
        while True:
            color, depth, intr = cam.grab(temporal_frames=1)  # 預覽用單幀就好
            # 即時把板畫出來，讓你確認姿態有偵測到
            got = detect_pose(color, intr, spec)
            disp = np.hstack((color, _depth_vis(depth)))
            ok_txt = f"pose OK (err={got[1]:.2f}px)" if got else "NO POSE (板沒被偵測到!)"
            col = (0, 255, 0) if got else (0, 0, 255)
            cv2.putText(disp, f"views: {len(records)}  |  {ok_txt}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.imshow("RealSense | Left: Color | Right: Depth", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                records = []; print("已清空。"); continue

            if key == ord("p"):
                # 真正拍攝時抓一串幀跑 temporal 濾波，深度乾淨很多
                color, depth, intr = cam.grab(temporal_frames=args.temporal_frames)
                got = detect_pose(color, intr, spec)
                if got is None:
                    print("拍攝失敗：板沒被偵測到（被物體/手擋住？太斜？太模糊？）")
                    continue
                pose, err = got
                roi = select_roi("Select Object ROI", color)
                if roi[2] <= 0 or roi[3] <= 0:
                    print("ROI 無效，取消。"); continue
                rec = save_view(debug_dir, len(records), color, depth, intr,
                                cam.depth_scale, roi, pose, err)
                records.append(rec)
                write_manifest(debug_dir, records, spec)  # 每拍一張就落地
                print(f"已擷取第 {len(records)} 個視角（reproj err={err:.2f}px）。")

            if key == ord("b"):
                if len(records) < args.min_views:
                    print(f"只有 {len(records)} 個視角，至少 {args.min_views} 個。"); continue
                write_manifest(debug_dir, records, spec)
                try:
                    # 關鍵：這裡呼叫的就是 reconstruct 指令用的同一條路
                    cmd_reconstruct(args, records=list(records), spec=spec)
                except RuntimeError as e:
                    print(f"建置失敗：{e}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


def _depth_vis(depth_u16):
    d = depth_u16.astype(np.float32)
    d = np.clip(d / (d.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


def build_parser():
    p = argparse.ArgumentParser(description="RealSense 轉盤掃描 -> MuJoCo 物件")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--assets", type=Path, default=Path("assets"))
        sp.add_argument("--board", choices=["charuco", "single"], default="charuco",
                        help="charuco=棋盤+多marker(強烈建議)，single=單張ArUco(你目前的做法)")
        sp.add_argument("--marker_dict", default="DICT_4X4_50")
        # charuco
        sp.add_argument("--squares_x", type=int, default=5)
        sp.add_argument("--squares_y", type=int, default=7)
        sp.add_argument("--square_length_m", type=float, default=0.030)
        sp.add_argument("--marker_length_m_charuco", type=float, default=0.022)
        # single
        sp.add_argument("--marker_id", type=int, default=0)
        sp.add_argument("--marker_length_m", type=float, default=0.037)

    def add_recon(sp):
        sp.add_argument("--min_views", type=int, default=8)
        sp.add_argument("--depth_trunc", type=float, default=1.0)
        sp.add_argument("--plate_z_clip_m", type=float, default=0.004)
        sp.add_argument("--mask_erode_px", type=int, default=3)
        # 物體包圍盒：只保留物體正上方那根圓柱範圍，砍掉放射狀刀片雜點
        sp.add_argument("--object_max_radius_m", type=float, default=0.08,
                        help="物體水平半徑上限(公尺)；離中心超過就砍。設 0 關閉")
        sp.add_argument("--object_max_height_m", type=float, default=0.20,
                        help="物體高度上限(公尺)；比轉盤面高過這值就砍。設 0 關閉")
        sp.add_argument("--flying_px_thresh_m", type=float, default=0.02,
                        help="飛點過濾門檻(公尺)；跟鄰居深度差超過就砍。設 0 關閉")
        sp.add_argument("--max_center_drift_m", type=float, default=0.02,
                        help="打滑偵測門檻(公尺)；單一視角物體重心偏離整體中心超過此值就"
                             "印警告(懷疑物體在轉盤上滑動了)。設很大的值可關閉此檢查")
        sp.add_argument("--tsdf_voxel_length", type=float, default=0.002)
        sp.add_argument("--tsdf_sdf_trunc", type=float, default=0.010)
        sp.add_argument("--max_visual_triangles", type=int, default=40000)
        sp.add_argument("--max_hulls", type=int, default=16)
        sp.add_argument("--max_neighbor_pose_jump_deg", type=float, default=60.0)
        sp.add_argument("--redetect", action="store_true",
                        help="離線重新偵測姿態（預設關；開了會讓結果可能跟拍攝當下不同）")
        sp.add_argument("--no_view", action="store_true")

    sp = sub.add_parser("make-board"); add_common(sp)
    sp.add_argument("--out", type=Path, default=Path("board.png"))

    sp = sub.add_parser("capture"); add_common(sp); add_recon(sp)
    sp.add_argument("--object_name", required=True)
    sp.add_argument("--width", type=int, default=848)
    sp.add_argument("--height", type=int, default=480)
    sp.add_argument("--fps", type=int, default=30)
    sp.add_argument("--preset", default="high_accuracy",
                    choices=["high_accuracy", "high_density", "default"])
    sp.add_argument("--temporal_frames", type=int, default=15,
                    help="每次拍攝抓幾幀跑 temporal 濾波（靜物拉高比較乾淨）")

    sp = sub.add_parser("reconstruct"); add_common(sp); add_recon(sp)
    sp.add_argument("--object_name", required=True)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "make-board":
        cmd_make_board(args)
    elif args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "reconstruct":
        cmd_reconstruct(args)


if __name__ == "__main__":
    main()