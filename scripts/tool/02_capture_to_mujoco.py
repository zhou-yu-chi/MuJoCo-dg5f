#!/usr/bin/env python3
"""
02_capture_to_mujoco.py
------------------------
物體放在轉盤上、轉盤貼一張 ArUco marker，繞著拍多個角度，用 marker 直接
讀出每一幀的精確旋轉，把深度影像融合成一個完整的封閉 mesh，再產生
MuJoCo 吃得下的物件資產（visual mesh + VHACD collision hulls +
object.xml），最後開 MuJoCo viewer 讓你立刻看到它掉在地上的樣子。

為什麼要用 marker，不是直接比對點雲：
這支腳本原本的做法是「徒手拍多角度 + 兩兩比對點雲特徵（FPFH+RANSAC+
ICP）自動算相對旋轉」，實測發現這對平面多、造型簡單的物體（方盒子、
罐子這類）非常不穩定——點雲特徵比對是靠局部曲率找對應點，平面/直角在
特徵空間裡到處長得一樣，很容易對錯角度，融合出來的 mesh 就會歪掉、
穿模。ArUco marker 不靠物體本身的幾何特徵，是相機直接讀 marker 在畫面
上的形變反推出精確旋轉角，跟物體是方的、圓的、有沒有花紋完全無關，是
目前對這類單機掃描最穩定的做法。

Pipeline：
  1. 物理設置：把物體放在一個可以轉動的轉盤（轉盤本身即可，不需要動
     力）上，轉盤上另外貼一張列印好的 ArUco marker（跟物體本身分開，
     不要被物體擋住），相機固定不動，每拍一張就手動把轉盤轉一個角度。
     因為物體、marker 都固定在轉盤上一起轉，marker 每一幀的姿態就等於
     物體那一幀的姿態，不需要另外量轉了幾度。
  2. 相機/分割：跟之前一樣沿用 01_cammer.py 的
     filter_object_depth_in_roi()，每一個角度都重新框 ROI、重新去背，
     因為物體轉了角度之後輪廓長得不一樣。
  3. 姿態：對同一張畫面跑 ArUco 偵測（cv2.aruco），用
     cv2.solvePnP 算出 marker 相對相機的精確位姿。因為 marker 貼在
     轉盤上、印刷面朝上，marker 自己的局部 Z 軸就是垂直向上，直接把
     marker 座標系當成 MuJoCo 的世界座標系（Z 朝上）用，不需要再猜相機
     有沒有水平拍攝、有沒有傾斜。
  4. 表面融合：把每一幀的物體深度影像，用該幀的相機姿態，直接做 TSDF
     volumetric fusion（ScalableTSDFVolume），比「先把點雲拼起來再做
     Poisson」乾淨很多，是原始深度資料逐像素融合。融合完用
     connected-components 只留最大的一塊，把邊緣雜訊/背景漏進來的小
     碎片丟掉。
  5. 之後：重心置中、重用 06_build_objects.py 的 VHACD 凸分解、重用
     08_try_grasp.py 的 build_single_object_scene() 產生完整場景、開
     viewer 預覽。

輸出位置：
  assets/objects/<object_name>/
    visual.stl                          TSDF 融合出的視覺 mesh
    collision/<name>_col_NN.stl         VHACD 碰撞 hull
    object.xml                          跟其他 YCB 物件同格式的 MJCF 片段
    capture_debug/                      每個視角的原始資料、mesh
  assets/_scene_<object_name>.xml       地板 + 手 + 這個新物件的完整場景

Usage:
    # 第一次用先印 marker（存成 PNG，直接印，不要讓印表機自動縮放）：
    python 02_capture_to_mujoco.py --make_marker marker.png --marker_id 0 --marker_length_m 0.05
    # 印出來後用尺量黑色方塊實際邊長，量出來跟 0.05 不一樣就用 --marker_length_m 改掉

    python 02_capture_to_mujoco.py --object_name my_cup --marker_length_m 0.05

操作方式：
    p : 拍攝一個視角（框物體的 ROI；轉一下轉盤再拍下一張，繞一圈拍
        8~16 張。畫面中必須同時看得到物體跟 marker，marker 被物體、
        手擋住會拍攝失敗）
    b : 用目前擷取到的所有視角融合 + 建置 MuJoCo 物件並預覽（建完視角
        不會清空，哪裡有破洞可以再補拍幾張、再按一次 b 重建）
    c : 清空目前擷取到的視角，重新開始拍下一個物件
    q : 離開
"""

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import trimesh
import mujoco
import mujoco.viewer

# --- 動態載入同專案內、檔名以數字開頭的既有腳本，直接重用其中的函式，
# 避免相機分割 / VHACD 分解 / 場景組裝這三段邏輯在兩個檔案裡各自維護、
# 之後互相漂移（做法跟 09_collect_grasps.py 動態載入 08_try_grasp.py 一致）。
_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent

_spec_cam = importlib.util.spec_from_file_location("cammer", _HERE / "01_cammer.py")
cam = importlib.util.module_from_spec(_spec_cam)
_spec_cam.loader.exec_module(cam)

_spec_bo = importlib.util.spec_from_file_location("build_objects", _SCRIPTS_DIR / "06_build_objects.py")
bo = importlib.util.module_from_spec(_spec_bo)
_spec_bo.loader.exec_module(bo)

_spec_tg = importlib.util.spec_from_file_location("try_grasp", _SCRIPTS_DIR / "08_try_grasp.py")
tg = importlib.util.module_from_spec(_spec_tg)
_spec_tg.loader.exec_module(tg)

# 跟 06_build_objects.py 的 geom 屬性完全一致：這幾個數字是刻意調過的，
# 讓物件跟手接觸時的軟硬度跟其他 YCB 物件一樣（見該檔案內的說明：如果用
# MuJoCo 引擎預設 solref，手跟物件接觸會取兩邊 solref 的平均，比手單獨
# 設定軟 2.5 倍，手指會明顯陷進物件表面）。
VISUAL_GEOM_ATTRS = 'contype="0" conaffinity="0" group="2" rgba="0.8 0.8 0.75 1"'
COLLISION_GEOM_ATTRS = (
    'group="3" condim="4" friction="0.8 0.01 0.001" density="1000" '
    'solref="0.005 1" solimp="0.9 0.98 0.001"'
)


@dataclass
class CapturedView:
    index: int
    color_image: np.ndarray              # 全畫面 BGR
    depth_masked: np.ndarray             # 全畫面 uint16，物體以外背景已歸零
    pose_cam_to_world: np.ndarray        # 4x4，這一幀相機座標系 -> marker(世界)座標系
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    roi: tuple
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)  # 相機畸變係數，rebuild 時要重新偵測 marker 姿態會用到


# ---------------------------------------------------------------------------
# ArUco marker：產生列印用的圖、從畫面讀出精確姿態
# ---------------------------------------------------------------------------

# marker 局部座標系（跟 OpenCV 舊版 estimatePoseSingleMarkers 內部用的定義
# 完全一致）：原點在 marker 中心，X 向右、Y 向上（沿著印刷面），Z 垂直穿出
# 印刷面。marker 平放在轉盤上、印刷面朝上時，這個 Z 軸在現實世界裡就是
# 「垂直向上」，所以直接拿 marker 座標系當 MuJoCo 世界座標系用，不用再猜
# 相機有沒有水平拍攝。
_MARKER_OBJECT_POINTS_UNIT = np.array([
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
], dtype=np.float64)


def make_marker_image(dictionary_name: str, marker_id: int, pixel_size: int = 800) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, pixel_size)
    # 外加一圈白色邊界（quiet zone），實務上對偵測穩定度有明顯幫助。
    margin = pixel_size // 6
    canvas = np.full((pixel_size + 2 * margin, pixel_size + 2 * margin), 255, dtype=np.uint8)
    canvas[margin:margin + pixel_size, margin:margin + pixel_size] = marker
    return canvas


def detect_marker_pose(
    color_image: np.ndarray,
    intr,
    dictionary_name: str,
    marker_id: int,
    marker_length_m: float,
    max_reprojection_error_px: float = 3.0,
) -> Optional[np.ndarray]:
    """
    在畫面中找 marker_id 這張 marker，回傳 4x4「相機座標系 -> marker(世界)
    座標系」的變換；找不到、或姿態不可靠就回傳 None。

    平面正方形 marker 用一般的 solvePnP（SOLVEPNP_ITERATIVE，只回傳一個
    解）在斜角看過去時有個常見的「翻轉歧義」：正確姿態跟鏡射過去的姿態
    投影誤差可能很接近，容易挑到錯的那個，融合出來的 mesh 會出現一根根
    從中心射出去的尖刺（那一幀整個被塞到錯的位置，沿著相機視線方向拖出
    一條）。改用 SOLVEPNP_IPPE_SQUARE（专門給方形 marker 設計）會一次回
    傳兩個候選解＋各自的重投影誤差，這裡直接挑誤差小的那個；誤差還是太
    大代表這一幀的偵測本身就不可靠（marker 太斜、太遠、太模糊），直接
    放棄整張，比硬用一個不可靠的姿態去污染整個融合結果好。
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None
    ids = ids.flatten()
    if marker_id not in ids:
        return None
    idx = int(np.where(ids == marker_id)[0][0])

    camera_matrix = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    object_points = _MARKER_OBJECT_POINTS_UNIT * (marker_length_m / 2.0)

    ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points, corners[idx], camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or len(rvecs) == 0:
        return None

    best = int(np.argmin(errors))
    if float(errors[best].item()) > max_reprojection_error_px:
        return None

    R, _ = cv2.Rodrigues(rvecs[best])
    marker_to_cam = np.eye(4)
    marker_to_cam[:3, :3] = R
    marker_to_cam[:3, 3] = tvecs[best].flatten()
    return np.linalg.inv(marker_to_cam)  # camera -> marker(world)


# ---------------------------------------------------------------------------
# TSDF 融合
# ---------------------------------------------------------------------------

def fuse_views_tsdf(
    views: list,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    small_cluster_ratio: float = 0.1,
) -> o3d.geometry.TriangleMesh:
    """用每一幀的 marker 姿態，把深度影像逐幀融合成一個封閉 mesh。"""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for v in views:
        color_o3d = o3d.geometry.Image(cv2.cvtColor(v.color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(v.depth_masked)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0 / v.depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(v.width, v.height, v.fx, v.fy, v.cx, v.cy)
        extrinsic = np.linalg.inv(v.pose_cam_to_world)  # world -> camera，integrate() 要的是這個方向
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF 融合後沒有產生任何三角形，請確認每個視角都有成功去背、深度沒有大片缺失")

    # TSDF 融合常會在物體邊緣留下背景漏進來的小碎片。原本只留「最大」的
    # 那一塊，但邊緣有時會黏出一片跟主體「連在一起」的插片（你圖上邊緣那些
    # 方形凸出），或飄出好幾塊中等大小的碎片。這裡改成：先算每一塊連通面的
    # 三角形數，凡是小於「最大塊 * small_cluster_ratio」的塊全部丟掉，一次
    # 清掉主體以外所有大大小小的碎片，只保留主體本身。
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_n = int(cluster_n_triangles.max())
    keep_cluster = cluster_n_triangles >= largest_n * small_cluster_ratio
    triangles_to_remove = np.logical_not(keep_cluster[triangle_clusters])
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    return mesh


def smooth_mesh(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int,
    method: str = "taubin",
) -> o3d.geometry.TriangleMesh:
    """
    深度相機在物體表面本來就有逐幀量測雜訊，TSDF 融合後表面會殘留一層淺淺
    的波浪起伏（不是尖刺，是緩緩的凹凸）。這裡對融合出來的 mesh 做網格平滑，
    把這層雜訊抹掉，讓光滑物體（耳機殼這類）看起來更貼近真實。

    method:
      "taubin"  —— Taubin 平滑。一般首選：它會交替做正向/反向平滑，能磨掉
                   高頻雜訊又「幾乎不收縮體積」，物體不會越磨越小。想要光滑
                   又不想失真時用這個。
      "laplacian" —— 標準 Laplacian 平滑，磨得更兇但會讓物體整體縮水、稜角
                   變鈍。只有在 taubin 還不夠平、且你不在意輕微縮水時才用。

    iterations 就是磨幾遍，越大越平滑也越鈍。0 代表完全不平滑（維持原樣）。
    """
    if iterations <= 0:
        return mesh
    if method == "laplacian":
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    else:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    mesh.compute_vertex_normals()
    return mesh


def cap_mesh_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    物體放在轉盤上掃描，貼著轉盤的那塊底面從頭到尾沒有任何一幀真正看到
    過（`clip_depth_below_plate` 還會再主動把貼轉盤的薄薄一圈砍掉避免
    轉盤本身混進 mesh），TSDF 融合出來的 mesh 在那裡一定會留一個洞，不是
    封閉的——這不是 bug，是單一轉盤掃描的物理限制（除非把物體整個翻過來
    再多掃一輪）。這裡找出邊界迴圈，直接扇形三角化補一塊平面蓋住洞口，
    等於假設看不到的那塊是平的，對「立在桌上/轉盤上的物體」這個假設通常
    合理，比留一個洞讓 collision mesh 穿模好非常多。
    """
    edges = mesh.edges_sorted
    edges_view = edges.view([("", edges.dtype)] * 2).ravel()
    _, inverse, counts = np.unique(edges_view, return_inverse=True, return_counts=True)
    boundary_edges = edges[counts[inverse] == 1]
    if len(boundary_edges) == 0:
        return mesh

    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited = set()
    new_vertices = list(mesh.vertices)
    new_faces = [tuple(f) for f in mesh.faces]

    for a0, b0 in boundary_edges:
        a0, b0 = int(a0), int(b0)
        start_key = (min(a0, b0), max(a0, b0))
        if start_key in visited:
            continue

        loop = [a0, b0]
        visited.add(start_key)
        closed = False
        for _ in range(len(boundary_edges) + 2):
            cur, prev = loop[-1], loop[-2]
            neighbors = [n for n in adjacency.get(cur, []) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt == a0:
                closed = True
                break
            edge_key = (min(cur, nxt), max(cur, nxt))
            if edge_key in visited:
                break
            visited.add(edge_key)
            loop.append(nxt)

        if not closed or len(loop) < 3:
            continue  # 形狀太怪的邊界（分岔、非流形）就放棄補這個洞，不硬補

        centroid = np.mean([mesh.vertices[i] for i in loop], axis=0)
        centroid_idx = len(new_vertices)
        new_vertices.append(centroid)
        for i in range(len(loop)):
            v0, v1 = loop[i], loop[(i + 1) % len(loop)]
            new_faces.append((v0, v1, centroid_idx))

    return trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces), process=True)


def build_object_only_scene(assets_dir: Path, safe_name: str) -> Path:
    """
    跟 08_try_grasp.py 的 build_single_object_scene() 不同：那個會 include
    dg5f_scene.xml，連手一起放進場景，適合手抓物件的既有 07/08/09 腳本用；
    這裡單純只要「拍出來的東西長什麼樣子」，開預覽不需要手，畫面乾淨很多。
    """
    scene_path = assets_dir / f"_preview_{safe_name}.xml"
    scene_path.write_text(
        f'<mujoco model="{safe_name}_preview">\n'
        '  <worldbody>\n'
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />\n'
        '    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 0" '
        'rgba="0.4 0.42 0.45 1" condim="3" friction="1.0 0.005 0.0001" />\n'
        '  </worldbody>\n'
        f'  <include file="objects/{safe_name}/object.xml" />\n'
        '</mujoco>\n'
    )
    return scene_path


def build_mujoco_object(
    tri_mesh: trimesh.Trimesh,
    obj_dir: Path,
    safe_name: str,
    max_hulls: int,
) -> tuple:
    """
    跟 06_build_objects.py 的 build_one() 做同一件事，差別只在來源是這次
    拍攝融合出的 in-memory mesh，不是硬碟上的 YCB nontextured.stl。
    重心平移到原點的規則、mesh 檔名規則、object.xml 的格式都刻意跟
    build_one() 一致，這樣 07/08/09 那些既有腳本可以直接把這個新物件當
    成另一個 YCB 物件使用。
    """
    com = tri_mesh.center_mass if tri_mesh.is_watertight else tri_mesh.centroid
    tri_mesh.apply_translation(-com)

    visual_path = obj_dir / "visual.stl"
    tri_mesh.export(visual_path)

    col_dir = obj_dir / "collision"
    col_dir.mkdir(exist_ok=True)
    # decompose() 讀的是 06_build_objects.py 模組內的 MAX_HULLS 全域常數，
    # 不是函式參數，這裡先蓋掉它才能讓 --max_hulls 真的生效。
    bo.MAX_HULLS = max_hulls
    hull_files = bo.decompose(tri_mesh, col_dir, safe_name)

    mesh_tags = [f'    <mesh name="{safe_name}_visual" file="objects/{obj_dir.name}/visual.stl" />']
    geom_tags = [f'      <geom type="mesh" mesh="{safe_name}_visual" {VISUAL_GEOM_ATTRS} />']
    for i, fn in enumerate(hull_files):
        mesh_name = f"{safe_name}_col_{i:02d}"
        mesh_tags.append(f'    <mesh name="{mesh_name}" file="objects/{obj_dir.name}/collision/{fn}" />')
        geom_tags.append(f'      <geom type="mesh" mesh="{mesh_name}" {COLLISION_GEOM_ATTRS} />')

    fragment = (
        "<mujoco>\n"
        "  <asset>\n" + "\n".join(mesh_tags) + "\n  </asset>\n"
        "  <worldbody>\n"
        f'    <body name="{safe_name}" pos="0 0 0.5">\n'
        f'      <freejoint name="{safe_name}_free" />\n'
        + "\n".join(geom_tags) + "\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    (obj_dir / "object.xml").write_text(fragment)

    info = {
        "watertight": bool(tri_mesh.is_watertight),
        "num_vertices": int(len(tri_mesh.vertices)),
        "num_triangles": int(len(tri_mesh.faces)),
        "num_collision_hulls": len(hull_files),
        "center_of_mass_before_recenter": [float(c) for c in com],
    }
    return fragment, info


def save_views_to_debug(obj_dir: Path, views: list) -> Path:
    """
    把目前拍到的每個視角（畫面、去背+侵蝕+砍轉盤後的深度、內參、marker
    姿態）存成檔案，這樣之後只想調融合/補洞/VHACD 這些「重建」參數時，
    可以直接用 --rebuild 重跑，不用重新架相機、重新繞著物體拍一輪。
    每次拍完都會整批重寫一次，不用擔心中途中斷資料不完整。
    """
    debug_dir = obj_dir / "capture_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for v in views:
        color_name = f"view_{v.index:02d}_color.png"
        depth_name = f"view_{v.index:02d}_depth_masked.png"
        cv2.imwrite(str(debug_dir / color_name), v.color_image)
        cam.save_uint16_png(debug_dir / depth_name, v.depth_masked)
        records.append({
            "index": v.index,
            "color_file": color_name,
            "depth_file": depth_name,
            "width": v.width,
            "height": v.height,
            "fx": v.fx,
            "fy": v.fy,
            "cx": v.cx,
            "cy": v.cy,
            "depth_scale": v.depth_scale,
            "roi": list(v.roi),
            "pose_cam_to_world": v.pose_cam_to_world.tolist(),
            "coeffs": list(v.coeffs),
        })
    views_json = debug_dir / "views.json"
    views_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return views_json


class _StaticIntrinsics:
    """detect_marker_pose() 只需要這幾個欄位，rebuild 時從存檔重建一個假的
    intrinsics 物件出來用，不需要真的 pyrealsense2 frame。"""

    def __init__(self, width, height, fx, fy, cx, cy, coeffs):
        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.ppx, self.ppy = cx, cy
        self.coeffs = list(coeffs)


def load_views_from_debug(
    obj_dir: Path,
    marker_dict: str = None,
    marker_id: int = None,
    marker_length_m: float = None,
    marker_max_reproj_err_px: float = 3.0,
    redetect_marker: bool = True,
) -> list:
    debug_dir = obj_dir / "capture_debug"
    views_json = debug_dir / "views.json"
    if not views_json.exists():
        raise RuntimeError(
            f"找不到 {views_json}，這個物件還沒有存過任何視角資料，沒辦法離線重建，"
            f"要先正常拍過至少一次（用 p 拍、按 b 建置或至少拍完一個視角）"
        )

    records = json.loads(views_json.read_text(encoding="utf-8"))
    views = []
    n_redetected, n_kept_old = 0, 0

    for r in records:
        color_image = cv2.imread(str(debug_dir / r["color_file"]))
        depth_masked = cv2.imread(str(debug_dir / r["depth_file"]), cv2.IMREAD_UNCHANGED)
        coeffs = tuple(r.get("coeffs", (0.0, 0.0, 0.0, 0.0, 0.0)))
        pose = np.array(r["pose_cam_to_world"], dtype=np.float64)

        # 存檔當下的姿態可能是用舊版演算法算的（例如平面 marker 常見的翻轉
        # 歧義，見 detect_marker_pose() 說明）；rebuild 時預設會拿存好的畫面
        # 重新偵測一次，用目前這版比較可靠的演算法更新姿態，這樣連姿態算
        # 錯的問題也能靠 --rebuild 修，不用重新架相機拍。
        if redetect_marker and marker_dict is not None:
            intr = _StaticIntrinsics(r["width"], r["height"], r["fx"], r["fy"], r["cx"], r["cy"], coeffs)
            fresh_pose = detect_marker_pose(
                color_image, intr, marker_dict, marker_id, marker_length_m,
                max_reprojection_error_px=marker_max_reproj_err_px,
            )
            if fresh_pose is not None:
                pose = fresh_pose
                n_redetected += 1
            else:
                n_kept_old += 1

        views.append(CapturedView(
            index=r["index"],
            color_image=color_image,
            depth_masked=depth_masked,
            pose_cam_to_world=pose,
            width=r["width"],
            height=r["height"],
            fx=r["fx"],
            fy=r["fy"],
            cx=r["cx"],
            cy=r["cy"],
            depth_scale=r["depth_scale"],
            roi=tuple(r["roi"]),
            coeffs=coeffs,
        ))

    if redetect_marker and marker_dict is not None:
        print(f"重新偵測 marker 姿態：{n_redetected} 個視角已更新"
              + (f"，{n_kept_old} 個視角重新偵測失敗、沿用舊姿態（這幾個角度的重建品質可能還是不可靠，建議之後補拍）"
                 if n_kept_old else ""))

    return views


def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def reject_pose_outliers(views: list, max_neighbor_jump_deg: float) -> list:
    """
    平面 marker 的翻轉歧義沒辦法只靠單一幀的重投影誤差百分之百擋掉：某些
    角度下，錯誤的鏡射解跟正確解投影誤差非常接近，連 detect_marker_pose()
    裡的 IPPE_SQUARE 挑最小誤差也可能挑到錯的那個。

    但物體是照拍攝順序一格一格繞著轉盤轉的，姿態理論上該平滑漸變——錯誤
    的那一幀姿態通常會跟拍攝順序上緊鄰的前後兩幀差非常多（往往剛好差了
    180 度），這裡用「跟前一幀、後一幀都差很多，但前一幀跳過它直接接
    後一幀卻很平順」這個訊號抓出這種孤立的壞幀，直接丟掉不參與融合。
    """
    if len(views) < 3:
        return views

    keep = [True] * len(views)
    for i in range(1, len(views) - 1):
        R_prev = views[i - 1].pose_cam_to_world[:3, :3]
        R_cur = views[i].pose_cam_to_world[:3, :3]
        R_next = views[i + 1].pose_cam_to_world[:3, :3]
        d_prev = _rotation_angle_deg(R_prev, R_cur)
        d_next = _rotation_angle_deg(R_cur, R_next)
        d_skip = _rotation_angle_deg(R_prev, R_next)
        if d_prev > max_neighbor_jump_deg and d_next > max_neighbor_jump_deg and d_skip < max_neighbor_jump_deg:
            keep[i] = False

    kept = [v for v, k in zip(views, keep) if k]
    dropped = [v.index for v, k in zip(views, keep) if not k]
    if dropped:
        print(f"[警告] 視角 {dropped} 的姿態跟拍攝順序上的前後鄰居差太多（疑似 marker 翻轉歧義），"
              f"已自動排除、不參與融合。如果丟太多張導致視角覆蓋不夠，考慮針對這幾個角度重拍。")
    return kept


def run_multiview_pipeline(views: list, args, obj_dir: Path, safe_name: str) -> dict:
    debug_dir = obj_dir / "capture_debug"
    save_views_to_debug(obj_dir, views)

    views = reject_pose_outliers(views, args.max_neighbor_pose_jump_deg)
    if len(views) < 3:
        raise RuntimeError("排除掉姿態異常的視角後剩不到 3 個，沒辦法融合，請重新拍攝並確保 marker 全程清楚可見")

    print("TSDF 融合中...")
    mesh_o3d = fuse_views_tsdf(
        views, args.tsdf_voxel_length, args.tsdf_sdf_trunc, args.depth_trunc,
        small_cluster_ratio=args.small_cluster_ratio,
    )
    o3d.io.write_triangle_mesh(str(debug_dir / "fused_mesh_world_frame.ply"), mesh_o3d)

    if args.smooth_iterations > 0:
        print(f"表面平滑中（{args.smooth_method}，{args.smooth_iterations} 遍）...")
        mesh_o3d = smooth_mesh(mesh_o3d, args.smooth_iterations, args.smooth_method)
        o3d.io.write_triangle_mesh(str(debug_dir / "fused_mesh_smoothed.ply"), mesh_o3d)

    if len(mesh_o3d.triangles) > args.max_visual_triangles:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=args.max_visual_triangles)

    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)

    if not tri_mesh.is_watertight:
        print("補洞中（底部貼轉盤那塊沒被掃到，補一塊平面蓋住）...")
        tri_mesh = cap_mesh_holes(tri_mesh)

    print(f"VHACD 凸分解中（最多 {args.max_hulls} 塊 collision hull）...")
    _, info = build_mujoco_object(tri_mesh, obj_dir, safe_name, args.max_hulls)

    meta = {
        "object_name": safe_name,
        "capture_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_views": len(views),
        "marker_dict": args.marker_dict,
        "marker_id": args.marker_id,
        "marker_length_m": args.marker_length_m,
        "tsdf_voxel_length": args.tsdf_voxel_length,
        "tsdf_sdf_trunc": args.tsdf_sdf_trunc,
        "mesh_info": info,
    }
    (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  watertight={info['watertight']} 三角形={info['num_triangles']} "
          f"collision hull 數={info['num_collision_hulls']}")

    return meta


def preview_in_mujoco(scene_path: Path):
    print(f"開啟 MuJoCo viewer 預覽：{scene_path}")
    print("（物件會從半空落下，關掉視窗就回到相機畫面）")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.002)


def select_roi_adjustable(window_name: str, image: np.ndarray) -> tuple:
    """
    跟 cv2.selectROI 一樣回傳 (x, y, w, h)，但可以先隨便拉一個大概的框，
    再個別抓左右/上下邊或四個角去微調，不用一次就拉到最終位置：
      - 在空白處拖拉：畫一個新框
      - 抓框的邊或角拖拉：只調整那一邊/角
      - 在框內部拖拉：整個框平移
    按 Enter 或空白鍵確認，按 c 或 Esc 取消（回傳全 0）。
    """
    handle_px = 10
    state = {"box": None, "drag": None, "last": None}

    def sorted_box(b):
        xa, ya, xb, yb = b
        return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)

    def hit_test(x, y, box):
        x0, y0, x1, y1 = sorted_box(box)
        near_left = abs(x - x0) <= handle_px
        near_right = abs(x - x1) <= handle_px
        near_top = abs(y - y0) <= handle_px
        near_bottom = abs(y - y1) <= handle_px
        inside_x = x0 - handle_px <= x <= x1 + handle_px
        inside_y = y0 - handle_px <= y <= y1 + handle_px
        if near_left and near_top:
            return "topleft"
        if near_right and near_top:
            return "topright"
        if near_left and near_bottom:
            return "bottomleft"
        if near_right and near_bottom:
            return "bottomright"
        if near_left and inside_y:
            return "left"
        if near_right and inside_y:
            return "right"
        if near_top and inside_x:
            return "top"
        if near_bottom and inside_x:
            return "bottom"
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "move"
        return None

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["box"] is None:
                state["box"] = [x, y, x, y]
                state["drag"] = "bottomright"
            else:
                handle = hit_test(x, y, state["box"])
                if handle is None:
                    state["box"] = [x, y, x, y]
                    handle = "bottomright"
                state["drag"] = handle
            state["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            xa, ya, xb, yb = state["box"]
            drag = state["drag"]
            if drag == "move":
                lx, ly = state["last"]
                dx, dy = x - lx, y - ly
                state["box"] = [xa + dx, ya + dy, xb + dx, yb + dy]
                state["last"] = (x, y)
                return
            if "left" in drag:
                xa = x
            if "right" in drag:
                xb = x
            if "top" in drag:
                ya = y
            if "bottom" in drag:
                yb = y
            state["box"] = [xa, ya, xb, yb]
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        disp = image.copy()
        if state["box"] is not None:
            x0, y0, x1, y1 = (int(v) for v in sorted_box(state["box"]))
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            handles = [
                (x0, y0), (x1, y0), (x0, y1), (x1, y1),
                ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1),
                (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2),
            ]
            for hx, hy in handles:
                cv2.circle(disp, (hx, hy), 4, (0, 255, 0), -1)
        cv2.putText(disp, "drag box/edges to adjust  ENTER=confirm  c/ESC=cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key in (27, ord("c")):  # Esc / c
            state["box"] = None
            break

    cv2.destroyWindow(window_name)
    if state["box"] is None:
        return (0, 0, 0, 0)
    x0, y0, x1, y1 = sorted_box(state["box"])
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def clip_depth_below_plate(
    depth_masked: np.ndarray,
    intr,
    depth_scale: float,
    pose_cam_to_world: np.ndarray,
    z_clip_m: float,
) -> np.ndarray:
    """
    ROI 框選/深度分布去背只能排除跟物體深度差夠多的背景，框太鬆或轉盤跟
    物體幾乎同高時，轉盤本身還是常常被當成物體一起留下來。這裡用已經算
    出來的 marker 姿態把每個像素反投影回世界座標，世界 Z（= 轉盤平面）
    以下/貼著轉盤的點直接砍掉，等於用「這個像素是不是浮在轉盤上方」再
    做一次跟物體形狀無關、對任何 ROI 鬆緊都有效的背景過濾。
    代價是物體貼著轉盤那薄薄一圈（z_clip_m 以內）也會被一起砍掉，通常
    可忽略；如果物體本身就有東西被誤砍，把 --plate_z_clip_m 調小即可。
    """
    ys, xs = np.nonzero(depth_masked)
    if len(xs) == 0:
        return depth_masked

    z_cam = depth_masked[ys, xs].astype(np.float64) * depth_scale
    x_cam = (xs - intr.ppx) / intr.fx * z_cam
    y_cam = (ys - intr.ppy) / intr.fy * z_cam
    pts_cam_h = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
    pts_world = pts_cam_h @ pose_cam_to_world.T

    below_plate = pts_world[:, 2] < z_clip_m
    out = depth_masked.copy()
    out[ys[below_plate], xs[below_plate]] = 0
    return out


def capture_one_view(
    color_image, depth_image, roi, intr, depth_scale, args, view_index: int
) -> CapturedView:
    pose = detect_marker_pose(
        color_image, intr, args.marker_dict, args.marker_id, args.marker_length_m,
        max_reprojection_error_px=args.marker_max_reproj_err_px,
    )
    if pose is None:
        raise RuntimeError(
            f"畫面中找不到 marker（dict={args.marker_dict}, id={args.marker_id}），或是姿態不可靠"
            f"（重投影誤差超過 {args.marker_max_reproj_err_px}px）。"
            f"請確認轉盤上的 marker 有進到畫面、沒被物體或手擋住、沒有太模糊、"
            f"盡量不要在太斜的角度看 marker（marker 看起來被壓得很扁時，姿態容易不準）"
        )

    result = cam.filter_object_depth_in_roi(
        color_image, depth_image, roi, intr, depth_scale,
        depth_trunc=args.depth_trunc, band_mm=args.object_band_mm,
    )
    cam.ensure_not_empty_pcd(result["object_pcd"], "拍攝到的物體點雲")

    x, y, w, h = roi
    depth_masked = np.zeros_like(depth_image, dtype=np.uint16)
    depth_masked[y:y + h, x:x + w] = result["masked_depth_roi"]

    if args.mask_erode_px > 0:
        # 深度相機在物體輪廓邊緣常見「飛點」：邊界像素的深度是物體跟背景
        # 插值出來的假值（RealSense 的 hole-filling 濾波器、GrabCut 的
        # 邊緣膨脹都會讓這圈假深度更容易被當成物體留下來），單一視角拍攝
        # 看不太出來，但很多視角一起融合，這圈假深度會疊成一圈貼著地板、
        # 往外攤開的薄殼/裙擺。往內侵蝕輪廓邊緣幾個像素可以把這圈雜訊去掉，
        # 拍的視角數夠多（8~16 個）時，邊緣被吃掉的部分會被其他角度補回來。
        kernel_size = 2 * args.mask_erode_px + 1
        eroded_mask = cv2.erode(result["full_mask"], np.ones((kernel_size, kernel_size), np.uint8))
        depth_masked[eroded_mask == 0] = 0

    depth_masked = clip_depth_below_plate(depth_masked, intr, depth_scale, pose, args.plate_z_clip_m)

    return CapturedView(
        index=view_index,
        color_image=color_image.copy(),
        depth_masked=depth_masked,
        pose_cam_to_world=pose,
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
        depth_scale=depth_scale,
        roi=(int(x), int(y), int(w), int(h)),
        coeffs=tuple(float(c) for c in intr.coeffs[:5]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str, help="物件名稱，會用來當資料夾名跟 MuJoCo body/mesh 名稱")
    parser.add_argument("--assets", type=Path, default=_SCRIPTS_DIR.parent / "assets")
    parser.add_argument("--make_marker", type=Path, default=None,
                         help="不開相機，只產生 marker PNG 到這個路徑然後結束（第一次用先跑這個）")
    parser.add_argument("--marker_dict", type=str, default="DICT_4X4_50",
                         help="ArUco 字典名稱，例如 DICT_4X4_50、DICT_5X5_100")
    parser.add_argument("--marker_id", type=int, default=0)
    parser.add_argument("--marker_length_m", type=float, default=0.05,
                         help="列印出來的 marker 黑色方塊實際邊長（公尺），務必用尺量過再填，量錯整個模型比例會跟著錯")
    parser.add_argument("--marker_max_reproj_err_px", type=float, default=3.0,
                         help="marker 姿態的重投影誤差門檻（像素），超過就視為這一幀偵測不可靠、直接拍攝失敗；"
                              "太嚴格常常拍不過可以調大，懷疑姿態不準（mesh 出現放射狀尖刺）可以調小")
    parser.add_argument("--max_neighbor_pose_jump_deg", type=float, default=60.0,
                         help="融合前檢查：跟拍攝順序上前後鄰居的姿態差超過這個角度（且跳過它前後能接得平順）"
                              "就視為翻轉歧義造成的壞幀，自動排除不參與融合；拍攝時故意每張都轉很大角度的話要調大")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--object_band_mm", type=int, default=60)
    parser.add_argument("--plate_z_clip_m", type=float, default=0.006,
                         help="世界座標系（= 轉盤平面）Z 低於這個值的像素視為轉盤本身，自動砍掉；"
                              "太小擋不住轉盤、太大會連物體貼轉盤的底部一起削掉")
    parser.add_argument("--mask_erode_px", type=int, default=4,
                         help="物體輪廓邊緣往內侵蝕幾個像素，去掉深度相機邊緣的飛點雜訊"
                              "（融合出來的 mesh 底部如果有一圈貼著地板往外攤開的薄殼/裙擺，調大這個值）")
    parser.add_argument("--tsdf_voxel_length", type=float, default=0.003, help="TSDF 融合體素大小（公尺）")
    parser.add_argument("--tsdf_sdf_trunc", type=float, default=0.015, help="TSDF 截斷距離（公尺）")
    parser.add_argument("--smooth_iterations", type=int, default=10,
                         help="融合後對 mesh 做幾遍網格平滑，抹掉深度相機殘留的表面波浪雜訊；"
                              "0=完全不平滑，光滑物體（耳機殼）建議 10~30，數字越大越平滑也越鈍。"
                              "rebuild 就會生效，可以反覆試")
    parser.add_argument("--smooth_method", type=str, default="taubin", choices=["taubin", "laplacian"],
                         help="平滑方法：taubin 幾乎不縮體積（首選）；laplacian 磨更兇但物體會縮水")
    parser.add_argument("--small_cluster_ratio", type=float, default=0.1,
                         help="融合後清碎片：連通面三角形數小於（最大塊 x 這個比例）的碎片全丟掉；"
                              "調大清得更兇（邊緣插片、飄出的小塊會被清掉），調太大可能連物體一部分都砍掉")
    parser.add_argument("--max_visual_triangles", type=int, default=40000)
    parser.add_argument("--max_hulls", type=int, default=bo.MAX_HULLS)
    parser.add_argument("--min_views", type=int, default=6, help="至少要有幾個視角才能建置（建議實際拍 8~16 個）")
    parser.add_argument("--no_view", action="store_true", help="建立完不要自動開 MuJoCo viewer 預覽")
    parser.add_argument("--preview_with_hand", action="store_true",
                         help="預覽視窗連手一起顯示（預設只顯示地板 + 你拍的物件，不含手）")
    parser.add_argument("--rebuild", action="store_true",
                         help="不開相機，直接用這個物件上次存好的 capture_debug/views.json 重新融合/"
                              "補洞/VHACD 建置，適合只想調整 --tsdf_voxel_length 等重建參數")
    args = parser.parse_args()

    if args.make_marker is not None:
        img = make_marker_image(args.marker_dict, args.marker_id, pixel_size=800)
        cv2.imwrite(str(args.make_marker), img)
        print(f"已產生 marker：{args.make_marker.resolve()}")
        print(f"dict={args.marker_dict} id={args.marker_id}")
        print("請直接列印這張圖（印表機設定選『實際大小 / 100%』，不要用『縮放至頁面』），")
        print("印出來後用尺量黑色方塊的邊長，實際量到的長度（公尺）記得用 --marker_length_m 帶進來，")
        print("跟這裡假設的不一樣模型比例會整個跑掉。")
        return

    if not args.object_name:
        parser.error("--object_name 是必填的（除非用 --make_marker 只印 marker）")

    safe_name = args.object_name.replace("-", "_")
    assets_dir = args.assets.resolve()
    obj_dir = assets_dir / "objects" / safe_name
    obj_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        # 不開相機：直接讀回上次拍過、存在 capture_debug/views.json 的視角
        # 資料重新跑融合/補洞/VHACD，適合只是想調 --tsdf_voxel_length、
        # --tsdf_sdf_trunc、--max_hulls 這類「重建」參數、不想重新繞著
        # 物體拍一輪的情況。--plate_z_clip_m、--mask_erode_px 這兩個是在
        # 拍攝當下就套用在存檔的深度資料上了，rebuild 不會重算，想改這兩
        # 個要重新拍。marker 姿態預設會拿存好的畫面重新偵測一次（見
        # load_views_from_debug 說明），連姿態算錯的問題也能靠這個修。
        views = load_views_from_debug(
            obj_dir,
            marker_dict=args.marker_dict,
            marker_id=args.marker_id,
            marker_length_m=args.marker_length_m,
            marker_max_reproj_err_px=args.marker_max_reproj_err_px,
        )
        print(f"從 {obj_dir / 'capture_debug' / 'views.json'} 讀回 {len(views)} 個視角，重新建置中...")
        try:
            run_multiview_pipeline(views, args, obj_dir, safe_name)
        except RuntimeError as e:
            print(f"建置失敗：{e}")
            return

        hand_scene_path = tg.build_single_object_scene(assets_dir, safe_name)
        preview_scene_path = build_object_only_scene(assets_dir, safe_name)
        print(f"\n=== 完成 ===\nobject.xml         = {obj_dir / 'object.xml'}")
        print(f"抓握用場景（含手） = {hand_scene_path}")
        print(f"預覽用場景（無手） = {preview_scene_path}\n")

        if not args.no_view:
            preview_in_mujoco(hand_scene_path if args.preview_with_hand else preview_scene_path)
        return

    pipeline = cam.rs.pipeline()
    config = cam.rs.config()
    config.enable_stream(cam.rs.stream.color, args.width, args.height, cam.rs.format.bgr8, args.fps)
    config.enable_stream(cam.rs.stream.depth, args.width, args.height, cam.rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = cam.rs.align(cam.rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    rs_filters = cam.create_realsense_post_filters()

    views: list = []

    print("\n=== 相機已開啟 ===")
    print(f"物件名稱：{safe_name}  ->  {obj_dir}")
    print(f"marker：dict={args.marker_dict} id={args.marker_id} 邊長={args.marker_length_m}m")
    print("操作方式：")
    print("  p : 拍攝一個視角（框物體 ROI；轉一下轉盤拍下一張，繞一圈 8~16 張）")
    print("  b : 用目前擷取到的視角融合 + 建置 MuJoCo 物件並預覽")
    print("  c : 清空目前擷取到的視角，重新開始")
    print("  q : 離開\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = cam.apply_realsense_post_filters(depth_frame, rs_filters)
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            stacked = np.hstack((color_image, cam.depth_to_vis(depth_image)))
            cv2.putText(stacked, f"views captured: {len(views)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("RealSense | Left: Color | Right: Depth", stacked)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                views = []
                print("已清空目前擷取到的視角。")
                continue

            if key == ord("p"):
                frozen_color = color_image.copy()
                frozen_depth = depth_image.copy()

                roi = select_roi_adjustable("Select Object ROI", frozen_color)
                x, y, w, h = roi
                if w <= 0 or h <= 0:
                    print("ROI 無效，取消這次拍攝。")
                    continue

                intr = color_frame.profile.as_video_stream_profile().intrinsics

                try:
                    view = capture_one_view(frozen_color, frozen_depth, roi, intr, depth_scale, args, len(views))
                except RuntimeError as e:
                    print(f"這個視角拍攝失敗：{e}")
                    continue

                views.append(view)
                save_views_to_debug(obj_dir, views)  # 立刻落地存檔，之後可以用 --rebuild 重跑不用重拍
                print(f"已擷取第 {len(views)} 個視角。")

            if key == ord("b"):
                if len(views) < args.min_views:
                    print(f"目前只有 {len(views)} 個視角，至少需要 {args.min_views} 個才能建置，請繼續按 p 拍攝。")
                    continue
                if len(views) < 8:
                    print(f"[提醒] 目前只有 {len(views)} 個視角，涵蓋角度可能不夠完整，建議 8~16 個效果較好。")

                try:
                    run_multiview_pipeline(views, args, obj_dir, safe_name)
                except RuntimeError as e:
                    print(f"建置失敗：{e}")
                    continue

                # _scene_<name>.xml（手 + 物件）留給 07/08/09 那些既有抓握腳本用；
                # _preview_<name>.xml（只有地板 + 物件）給這裡的預覽用，畫面乾淨。
                hand_scene_path = tg.build_single_object_scene(assets_dir, safe_name)
                preview_scene_path = build_object_only_scene(assets_dir, safe_name)
                print(f"\n=== 完成 ===\nobject.xml         = {obj_dir / 'object.xml'}")
                print(f"抓握用場景（含手） = {hand_scene_path}")
                print(f"預覽用場景（無手） = {preview_scene_path}\n")

                if not args.no_view:
                    preview_in_mujoco(hand_scene_path if args.preview_with_hand else preview_scene_path)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()