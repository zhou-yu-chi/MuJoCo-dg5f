#!/usr/bin/env python3
"""
08_try_grasp.py
----------------
Stage-3 core, single attempt: implements plan.md's try_grasp() against one
YCB object, as a debuggable/visualizable script before it gets wrapped in
a multiprocess batch collector (next script).

One deliberate deviation from plan.md's pseudocode: it computes
`palm_pos = p - n * d`, but trimesh face_normals point *outward* from the
mesh surface, so `p - n*d` walks *into* the object, not away from it. That
contradicts "手掌...退開" (palm backs away). This script uses the
physically-correct `p + n*d` (retreat along the outward normal) and then
approaches along `approach_dir` (see below -- not simply `-n` anymore).

Pipeline for one attempt:
  1. Object free-falls onto the floor and settles (as in 07) to get a
     real resting pose; the hand holds a static PREGRASP shape throughout
     so it doesn't disturb anything before the attempt starts.
  2. sample_grasp_point() picks point `p` + outward normal `n` on the
     object's *visual* mesh (not the convex collision hulls -- see
     06_build_objects.py for why) via antipodal sampling, not plain
     uniform-random: candidates are rejected unless a ray cast from `p`
     along `-n` hits an opposing wall within the hand's graspable width
     and roughly antipodal to `n`, and unless `p` clears the floor by a
     margin. `approach_dir` is `-n` blended slightly toward the object's
     centroid (CENTROID_BLEND) so the palm aims into the bulk of the
     object rather than skimming its surface tangentially.
  3. Palm is teleported (qpos write, no physics) to `p + approach_dir*-d`
     i.e. retreated by `d` along `-approach_dir` (d ~ U(0.19, 0.26)),
     oriented so its local +Z (== the palm_center site direction, i.e.
     "toward the fingers") points along `approach_dir`, plus a random
     roll about that axis for wrist variety.
  4. Fingers pre-shape to PREGRASP + noise.
  5. Palm is walked forward in small kinematic steps along `approach_dir`
     until any hand geom touches the object, or the hand hits the floor
     first (bad sample, bail out), or the budget `d` runs out with no
     contact -- antipodal filtering cuts down on outright misses, but
     bailing out here early still matters for throughput once batched.
  6. Finger actuators are commanded to CLOSE_FRACTION and the sim runs
     under physics (palm is no longer teleported, real dynamics take
     over) with the object's gravity compensated, so the fingers can
     finish forming the grasp before gravity fights them.
  7. Object gravcomp is turned back off (real gravity) and 1N pushes are
     applied along +-x/+-y/+-z for 1s each; success = displacement from
     the pre-push pose stays under 2cm through every push.

Usage:
    python 08_try_grasp.py --assets ../assets --object 025_mug --viewer
    python 08_try_grasp.py --assets ../assets --object 025_mug --seed 3
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

FINGERS = {
    "thumb":  [f"lj_dg_1_{k}" for k in range(1, 5)],
    "index":  [f"lj_dg_2_{k}" for k in range(1, 5)],
    "middle": [f"lj_dg_3_{k}" for k in range(1, 5)],
    "ring":   [f"lj_dg_4_{k}" for k in range(1, 5)],
    "pinky":  [f"lj_dg_5_{k}" for k in range(1, 5)],
}
PALM_HEIGHT = 0.0738  # palm origin -> palm_center site, see 03_build_scene_v3.py

PREGRASP_FRACTION = 0.25
PREGRASP_NOISE_STD = 0.1
CLOSE_FRACTION = 0.85

# Must clear the pregrasp fingertip reach along the approach axis (~0.161m
# from palm_center at PREGRASP_FRACTION, measured with the fixed
# flexion_open_close semantics -- see forward-kinematics check done while
# fixing this) or the fingers already overlap the object at spawn, before
# the approach loop takes a single step. Old values (0.08~0.15) were
# below that reach, which is why "contact at approach step 0" fired
# immediately on essentially every attempt.
D_MIN, D_MAX = 0.19, 0.26
N_APPROACH_STEPS = 150

# 逼近階段「碰到就停」的安全煞車，改成看穿透量/接觸點數，而不是有碰到就停。
# 原因：預張開時五指伸出的長度不一（中指最長、大拇指最短），逼近時最先碰
# 到物體的常常只是「跑最遠的那根手指」自己先擦到，如果一碰到就整個停止
# 逼近，手掌跟其他比較短的手指根本還沒進到能碰到物體的範圍內，收攏時自
# 然只有那一根手指鎖得住、其餘手指全部撲空——這正是先前很多次抓取只鎖到
# 一根手指、拿起測試 0cm 的主因。改成只要還沒撞出明顯穿透/沒撞到一大堆
# 點，就讓手掌繼續往取樣點靠，直到真的撞太深、或走到取樣點本身為止。
MAX_APPROACH_PENETRATION_M = 0.01   # 逼近中允許的最大穿透深度
MAX_APPROACH_CONTACTS = 30          # 逼近中允許的最大同時接觸點數
N_CLOSE_STEPS = 800
N_POSTCLOSE_SETTLE = 300
LIFT_HEIGHT = 0.10          # lift test: raise the palm 10cm, object must follow
N_LIFT_STEPS = 500
LIFT_SUCCESS_FRACTION = 0.5  # object must rise >= 50% of LIFT_HEIGHT to count
N_POSTLIFT_SETTLE = 300
N_PUSH_STEPS = 500          # 1s at timestep=0.002
PUSH_FORCE_N = 1.0
SUCCESS_DISP_M = 0.02

DROP_POS = np.array([0.0, 0.0, 0.15])
N_DROP_SETTLE = 1500

# 取樣點篩選（對蹠點取樣，antipodal sampling）：純粹在表面均勻隨機取一點，
# 完全不管對面有沒有東西可以夾、夾的寬度合不合理，是抓取成功率趨近於 0
# 的主因之一。做法：一次批次取 N_SAMPLE_CANDIDATES 個候選點，對每個點沿著
# 內法向量往物體內部打一條射線，找到對面那道「牆」——如果牆的法向量跟候選
# 點大致相反、而且兩者間距落在手掌環抱得住的寬度範圍內，才是合格的候選。
N_SAMPLE_CANDIDATES = 300
MIN_GRASP_WIDTH_M = 0.02     # 太薄（邊緣、把手邊邊）夾不到什麼東西
MAX_GRASP_WIDTH_M = 0.12     # 對應手指展開後大約能環抱的直徑，太粗根本包不住
ANTIPODAL_COS_THRESH = -0.5  # 兩邊法向量夾角要大於 120 度，才算真的「對面有牆」
FLOOR_CLEARANCE_M = 0.03     # 候選點世界座標高度至少要離地板這麼遠
CENTROID_BLEND = 0.2         # 逼近方向朝物體重心微調的混合權重（0=完全用表面法向量）

# 收攏階段安全煞車：物體在收攏中被 gravcomp 設成無重力（見 phase 4），手掌
# 也完全沒有釘住（讓接觸反作用力可以自然把手掌帶回貼合的姿勢）。當取樣到
# 不好的逼近角度、一開始就疊了大量重疊接觸時，兩邊都沒有重量可以把自己拉
# 回原位，解算器硬解開重疊的力道會把手跟物體一起彈飛（曾經量到彈到 2 公尺
# 高、物體最終落點離原位 1 公尺遠）。這兩個閾值讓我們在爆衝的第一時間就
# 中止收攏，而不是放任它們在空中糾纏完剩下幾百步才在最後才判定失敗。
MAX_OBJ_SPEED_MPS = 2.0     # 物體瞬時速度超過這個值，基本上就是被彈飛，不是手指正常推擠
DIVERGE_DISP_M = 0.5        # 物體位移超過這個值，判定抓取已經失控

# 手指「一碰到物體就鎖定角度、不再繼續收攏」的機制，原本鎖了就不會再解開；
# 但如果那次接觸只是逼近／收攏過程中的一次擦過（例如收攏初期的重疊接觸被
# 解算器推開後，該手指其實根本沒真的貼住物體），永久鎖定會讓那根手指停在
# 半彎的角度、看起來像「沒有做抓握動作」。連續這麼多步都偵測不到接觸，就
# 視為那次接觸只是擦過去，解除鎖定、讓它繼續往收攏目標靠攏。
UNLOCK_AFTER_STEPS = 30


def build_single_object_scene(assets_dir: Path, obj_name: str) -> Path:
    safe = obj_name.replace("-", "_")
    scene_path = assets_dir / f"_scene_{safe}.xml"
    scene_path.write_text(
        "<mujoco>\n"
        '  <include file="dg5f_scene.xml" />\n'
        f'  <include file="objects/{obj_name}/object.xml" />\n'
        "</mujoco>\n"
    )
    return scene_path


def joint_index(model):
    idx = {}
    for jnames in FINGERS.values():
        for jn in jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}")
            idx[jn] = (jid, aid)
    return idx


# Per-joint (open_val, close_val) for a true flexion joint, or None if the
# joint should be held neutral (spread/rotation, not flexion). Verified
# against Tesollo's own Manus-glove teleop driver for the real hand
# (dg5f_driver/script/manus_retarget/manus_retarget.py: dir_arr, PLUS the
# mQd post-processing sign clamp -- the sign clamp is the important part,
# since it reveals which HALF of a joint's range the vendor's own driver
# actually uses).
#
# - Every finger's "_1" (thumb's CMC ab-adduction; index/middle/ring's
#   MCP ab-adduction) is spread, axis x, held neutral.
# - Index/middle/ring's "_2" (MCP flexion, axis y) has an asymmetric
#   range starting at 0, so lo==0 is already "open" -- sweeps lo -> hi.
# - Thumb's "_2" is a big opposition rotation (axis z, range (0, pi)),
#   also lo==0==open -- sweeps lo -> hi (mQd[1] clamped >=0 confirms).
# - Every finger's "_3"/"_4" (PIP/DIP, axis y, x for thumb) share the
#   SYMMETRIC range (-pi/2, pi/2). The vendor driver's post-processing
#   clamps these to only ONE HALF of that range (mQd never crosses 0),
#   so "open" is qpos=0 (straight finger), NOT lo or hi -- sweeping the
#   full lo->hi range (as an earlier version of this function did) starts
#   the "open" pose hyperextended ~45deg backwards, which is what made
#   pregrasp/close look like a weird/broken pose. Non-thumb closes toward
#   +pi/2; thumb closes toward -pi/2.
# - Pinky is NOT structurally the same as the other three fingers: its
#   "_1" (axis z, range (-1.05, 0.02)) is a unique palm-arch/opposition
#   rotation no other finger has, and its "_2" (axis x, range
#   (-0.61, 0.42) -- identical to index/middle/ring's "_1") is actually
#   the spread/abduction joint, NOT an MCP flexion joint. The vendor
#   driver skips BOTH lj_dg_5_1 and lj_dg_5_2 in its sign-clamp
#   post-processing (drives them straight off raw spread ergonomics,
#   same as index/middle/ring's "_1"), confirming both are spread-like
#   and must be held neutral, not swept as flexion -- sweeping pinky's
#   "_2" as if it were an MCP joint (previous bug) swings the whole
#   pinky sideways instead of curling it. Only pinky's "_3"/"_4" are
#   true flexion joints.
def flexion_open_close(jn: str, lo: float, hi: float):
    finger, stage = (int(x) for x in jn.split("_")[-2:])
    if stage == 1:
        return None
    if finger == 5 and stage == 2:
        return None
    if stage == 2:
        return (lo, hi)
    return (0.0, lo) if finger == 1 else (0.0, hi)


def hand_pose_ctrl(model, idx, fraction, rng=None, noise_std=0.0):
    """ctrl vector: spread/rotation joints held neutral, true flexion
    joints swept `fraction` of the way from open toward the verified
    close side (see flexion_open_close)."""
    ctrl = np.zeros(model.nu)
    for jn, (jid, aid) in idx.items():
        lo, hi = model.jnt_range[jid]
        bounds = flexion_open_close(jn, lo, hi)
        val = 0.0 if bounds is None else bounds[0] + fraction * (bounds[1] - bounds[0])
        if noise_std > 0 and rng is not None:
            val += rng.normal(0.0, noise_std)
        ctrl[aid] = float(np.clip(val, lo, hi))
    return ctrl


def sample_grasp_point(mesh, obj_pos: np.ndarray, obj_rot: np.ndarray, rng: np.random.Generator):
    """在物體表面找一個「對面真的有東西可以夾」的取樣點，取代單純均勻隨機
    取一個表面點。

    做法是傳統幾何抓取法裡最便宜的一招——對蹠點取樣（antipodal
    sampling）：批次取一堆候選點，對每個點沿著內法向量往物體內部打一條
    射線，找到對面那道「牆」。只有當牆的法向量跟候選點的法向量大致相反
    （代表真的是兩片相對的表面，不是切線擦過去），而且兩者間距落在手掌
    環抱得住的寬度範圍內，才算合格候選；還會濾掉太貼近地板的點（逼近時
    很容易撞地板，見 hand_touches_floor）。純隨機取樣完全不管這些，是先
    前抓取成功率趨近於 0 的主因之一。

    順便：逼近方向不是死板地沿著表面法向量走，而是朝物體重心的方向微調
    一點（CENTROID_BLEND），讓手掌更容易對準物體「中心厚實的地方」，而不
    是貼著表面切線滑過去。

    回傳 (p_world, n_world, approach_dir)；找不到合格候選點時（例如形狀
    太不規則，如香蕉的彎曲側面）退回單純隨機取一點，不會讓整次嘗試卡死。
    """
    seed = int(rng.integers(0, 2**31 - 1))  # 順便修掉 trimesh 取樣不吃 rng 的可重現性問題
    p_local, face_idx = trimesh.sample.sample_surface(mesh, N_SAMPLE_CANDIDATES, seed=seed)
    n_local = mesh.face_normals[face_idx]

    eps = 1e-4  # 起點稍微往內縮一點，避免射線一開始就打到自己所在的那個面
    origins = p_local - n_local * eps
    directions = -n_local
    locations, index_ray, index_tri = mesh.ray.intersects_location(origins, directions)

    # 同一條射線可能穿過好幾個面，只留「最近」的那個當作對面的牆
    best_dist, best_tri = {}, {}
    for loc, ray_i, tri_i in zip(locations, index_ray, index_tri):
        dist = float(np.linalg.norm(loc - origins[ray_i]))
        if ray_i not in best_dist or dist < best_dist[ray_i]:
            best_dist[ray_i] = dist
            best_tri[ray_i] = tri_i

    centroid_world = obj_pos + obj_rot @ mesh.center_mass

    def build(i: int):
        p_w = obj_pos + obj_rot @ p_local[i]
        n_w = obj_rot @ n_local[i]
        n_w = n_w / np.linalg.norm(n_w)
        to_centroid = centroid_world - p_w
        to_centroid = to_centroid / (np.linalg.norm(to_centroid) + 1e-9)
        approach_dir = (1 - CENTROID_BLEND) * (-n_w) + CENTROID_BLEND * to_centroid
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        return p_w, n_w, approach_dir

    valid = []
    for ray_i, tri_i in best_tri.items():
        width = best_dist[ray_i]
        if not (MIN_GRASP_WIDTH_M <= width <= MAX_GRASP_WIDTH_M):
            continue
        if np.dot(n_local[ray_i], mesh.face_normals[tri_i]) >= ANTIPODAL_COS_THRESH:
            continue
        p_w, _, _ = build(ray_i)
        if p_w[2] < FLOOR_CLEARANCE_M:
            continue
        valid.append(ray_i)

    if valid:
        return build(int(rng.choice(valid)))

    print("[diag] no antipodal candidate found, falling back to a plain random surface point")
    return build(int(rng.integers(0, N_SAMPLE_CANDIDATES)))


def rotation_local_z_to(direction: np.ndarray, roll: float) -> Rotation:
    """Rotation R such that R @ [0,0,1] == direction, with an extra `roll`
    radians about that same axis (applied in the local frame first)."""
    direction = direction / np.linalg.norm(direction)
    align, _ = Rotation.align_vectors([direction], [[0.0, 0.0, 1.0]])
    return align * Rotation.from_rotvec([0.0, 0.0, roll])


def hand_touches_object(model, data, obj_body_id) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 == obj_body_id and b2 != 0:
            return True
        if b2 == obj_body_id and b1 != 0:
            return True
    return False


def hand_touches_floor(model, data, floor_gid: int, obj_body_id: int) -> bool:
    """手（不是物體本身，物體貼在地板上是正常狀態）是否碰到地板。用來在
    逼近階段擋掉「取樣點/角度太貼近地板」的取樣結果——這種取樣會讓手掌沿
    著逼近方向走到一半就整個插進地板，收攏階段一開始就疊出大量重疊接觸，
    是「手跟物體一起被彈飛」的主要成因之一。"""
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        other = None
        if c.geom1 == floor_gid:
            other = b2
        elif c.geom2 == floor_gid:
            other = b1
        if other is not None and other not in (0, obj_body_id):
            return True
    return False


def step(model, data, viewer=None):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
        time.sleep(0.002)


def run(model, data, obj_name: str, rng: np.random.Generator, viewer=None) -> dict:
    """跑一次完整的抓取嘗試，回傳這次嘗試的結果字典（給批次收集腳本用）：

        {
          "object": obj_name,
          "success": bool,
          "fail_reason": str | None,   # 失敗時是哪個階段擋下來的；成功時是 None
          "sample_p": [x,y,z] | None,  # 取樣到的抓取點（世界座標）
          "sample_n": [x,y,z] | None,  # 該點法向量
          "object_pose": {"pos": [...], "quat": [wxyz]} | None,  # 只有成功才填
          "palm_pose":   {"pos": [...], "quat": [wxyz]} | None,  # 抓取當下、推力測試前
          "joint_angles": [...20 個手指關節角 (rad)...] | None,
        }

    後三個欄位只有 success=True 時才會填，格式對應 doc/plan.md 階段 3 要求
    的訓練資料（object_pose／palm_pose／joint_angles）。
    """
    result = {
        "object": obj_name,
        "success": False,
        "fail_reason": None,
        "sample_p": None,
        "sample_n": None,
        "object_pose": None,
        "palm_pose": None,
        "joint_angles": None,
    }

    idx = joint_index(model)
    pregrasp_ctrl = hand_pose_ctrl(model, idx, PREGRASP_FRACTION, rng, PREGRASP_NOISE_STD)
    close_ctrl = hand_pose_ctrl(model, idx, CLOSE_FRACTION)

    safe = obj_name.replace("-", "_")
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, safe)
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{safe}_free")
    obj_qadr = model.jnt_qposadr[obj_jid]
    obj_vadr = model.jnt_dofadr[obj_jid]
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
    palm_qadr = model.jnt_qposadr[free_jid]
    palm_vadr = model.jnt_dofadr[free_jid]
    palm_spawn_qpos = data.qpos[palm_qadr:palm_qadr + 7].copy()

    # hand holds its own weight the whole time (stand-in for a wrist/arm
    # that isn't simulated yet); object starts under real gravity.
    for bid in range(model.nbody):
        model.body_gravcomp[bid] = 0.0 if bid == obj_bid else 1.0

    # ---- phase 1: drop the object, let it settle -----------------------
    # Palm is pinned at its spawn pose here (qpos reset + qvel zeroed every
    # step) for the same reason as 04_test_close.py: the fingers snapping
    # from qpos=0 into PREGRASP on the very first step kicks a reaction
    # impulse into the free-floating palm. Without pinning, that leftover
    # velocity survives (unnoticed, since phase 3 overwrites qpos every
    # step and masks it) until phase 4 hands control to real physics, at
    # which point it launches the hand -- this is what "手掉到馬克杯上面
    # 然後一起消失" was: stale qvel finally taking effect.
    data.qpos[obj_qadr:obj_qadr + 3] = DROP_POS
    data.qpos[obj_qadr + 3:obj_qadr + 7] = [1, 0, 0, 0]
    data.ctrl[:] = pregrasp_ctrl
    for _ in range(N_DROP_SETTLE):
        step(model, data, viewer)
        data.qpos[palm_qadr:palm_qadr + 7] = palm_spawn_qpos
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    obj_pos = data.xpos[obj_bid].copy()
    obj_rot = data.xmat[obj_bid].reshape(3, 3).copy()
    print(f"object settled at {obj_pos.round(3)}")

    # ---- phase 2: 用對蹠點取樣找一個「環抱得住」的表面點 -------------------
    mesh = trimesh.load(Path(__file__).resolve().parents[1]
                         / "assets" / "objects" / obj_name / "visual.stl", force="mesh")
    p_world, n_world, approach_dir = sample_grasp_point(mesh, obj_pos, obj_rot, rng)
    result["sample_p"] = p_world.tolist()
    result["sample_n"] = n_world.tolist()

    d = rng.uniform(D_MIN, D_MAX)
    roll = rng.uniform(0.0, 2 * np.pi)
    print(f"sampled p={p_world.round(3)} n={n_world.round(3)} d={d:.3f} roll={np.degrees(roll):.0f}deg")

    # 手掌姿態跟著 approach_dir 走（已經混入朝重心微調的方向，不是死板的
    # -n_world），逼近路徑的位置也要用同一條軸，兩者才會一致。
    R = rotation_local_z_to(approach_dir, roll)
    quat_xyzw = R.as_quat()
    palm_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    R_mat = R.as_matrix()

    def set_palm(offset):
        # offset=d 時退到最外側，offset=0 時剛好落在取樣點 p_world 上
        center_pos = p_world - approach_dir * offset
        body_pos = center_pos - R_mat @ np.array([0.0, 0.0, PALM_HEIGHT])
        data.qpos[palm_qadr:palm_qadr + 3] = body_pos
        data.qpos[palm_qadr + 3:palm_qadr + 7] = palm_quat
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    # ---- phase 3: kinematic approach until contact ----------------------
    # set_palm() only overwrites qpos; mj_step still integrates qvel every
    # step from whatever forces act during that step (contact, actuator),
    # so qvel is zeroed on every iteration too -- otherwise it accumulates
    # silently under the qpos overwrite and only shows up once phase 4
    # hands control to real physics (see phase-1 comment above).
    data.ctrl[:] = pregrasp_ctrl
    contact_step = None
    for i in range(N_APPROACH_STEPS + 1):
        offset = d * (1.0 - i / N_APPROACH_STEPS)
        set_palm(offset)
        step(model, data, viewer)
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0
        # 取樣點/法向量太貼近地板時，手掌沿著逼近方向走到一半，還沒真的碰到
        # 物體，指節或手掌本身就已經插進地板——這種取樣點直接判失敗，不要讓
        # 它帶著一身地板穿透量進入收攏階段（那是後面「手跟物體一起被彈飛」
        # 的主要成因之一，見 MAX_OBJ_SPEED_MPS/DIVERGE_DISP_M 旁的說明）。
        if hand_touches_floor(model, data, floor_gid, obj_bid):
            print("FAIL: hand hit the floor while approaching (sampled point/angle too close to the floor)")
            result["fail_reason"] = "hit_floor"
            return result
        if hand_touches_object(model, data, obj_bid):
            contact_step = i
            # 不是一碰到就整個停止逼近：只要目前的接觸還不算「撞太深/撞
            # 太多點」，就讓手掌繼續往取樣點靠，好讓比較短的手指跟手掌本
            # 身也有機會進到能碰到物體的範圍內，而不是被最先擦到的那根
            # 手指卡住（見上面 MAX_APPROACH_PENETRATION_M 的說明）。
            pen = min(
                (c.dist for c in data.contact[:data.ncon]
                 if obj_bid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])),
                default=0.0,
            )
            if -pen > MAX_APPROACH_PENETRATION_M or data.ncon > MAX_APPROACH_CONTACTS:
                break

    if contact_step is None:
        print("FAIL: no contact within approach budget")
        result["fail_reason"] = "no_contact"
        return result
    print(f"contact at approach step {contact_step}/{N_APPROACH_STEPS}")

    # ---- phase 4: close fingers gradually, each stops on its own contact -
    # Palm is left fully physics-driven here (not pinned) so contact
    # reaction forces from the closing fingers can naturally reseat it,
    # same as a real wrist would give slightly under finger contact.
    #
    # Snapping every joint straight to CLOSE_FRACTION the instant contact
    # is seen (the old behaviour) drives all 20 joints at full torque
    # toward a deep target regardless of what they've already hit -- on a
    # small, light object that punches it away rather than wrapping it,
    # and whatever doesn't get punched away gets driven through the mesh.
    # Instead: ramp the ctrl target gradually finger-by-finger, and freeze
    # a finger's target the moment any part of it touches the object, so
    # each finger stops with just enough force to hold contact instead of
    # continuing to drive through it.
    finger_body_ids = {
        fname: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ll_dg_{i}_{k}") for k in range(1, 5)]
        for i, fname in enumerate(FINGERS.keys(), start=1)
    }
    locked = {fname: False for fname in FINGERS}
    miss_streak = {fname: 0 for fname in FINGERS}  # 鎖定後，連續幾步偵測不到接觸
    current_ctrl = pregrasp_ctrl.copy()
    model.body_gravcomp[obj_bid] = 1.0
    diverged = False
    for step_i in range(N_CLOSE_STEPS):
        frac = min(1.0, (step_i + 1) / N_CLOSE_STEPS)
        for fname, jnames in FINGERS.items():
            touched = any(
                (model.geom_bodyid[c.geom1] == obj_bid and model.geom_bodyid[c.geom2] in finger_body_ids[fname])
                or (model.geom_bodyid[c.geom2] == obj_bid and model.geom_bodyid[c.geom1] in finger_body_ids[fname])
                for c in data.contact[:data.ncon]
            )
            if locked[fname]:
                if touched:
                    miss_streak[fname] = 0
                else:
                    miss_streak[fname] += 1
                    if miss_streak[fname] >= UNLOCK_AFTER_STEPS:
                        # 鎖定之後一直偵測不到接觸，代表當初那次接觸只是擦過去，
                        # 不是真的貼住物體——解除鎖定，讓它繼續往收攏目標靠攏。
                        locked[fname] = False
                if locked[fname]:
                    continue
            elif touched:
                locked[fname] = True
                miss_streak[fname] = 0
                continue
            for jn in jnames:
                _jid, aid = idx[jn]
                current_ctrl[aid] = pregrasp_ctrl[aid] + frac * (close_ctrl[aid] - pregrasp_ctrl[aid])
        data.ctrl[:] = current_ctrl
        step(model, data, viewer)

        # 安全煞車：物體被單一步的重疊接觸力瞬間彈飛時，速度或位移會在幾步
        # 內就爆衝超標，這裡當場中止收攏，不要放任手跟物體繼續在空中糾纏
        # 剩下的幾百步（見 MAX_OBJ_SPEED_MPS/DIVERGE_DISP_M 的說明）。
        obj_speed = np.linalg.norm(data.qvel[obj_vadr:obj_vadr + 3])
        obj_disp = np.linalg.norm(data.qpos[obj_qadr:obj_qadr + 3] - obj_pos)
        if not np.all(np.isfinite(data.qpos)) or obj_speed > MAX_OBJ_SPEED_MPS or obj_disp > DIVERGE_DISP_M:
            diverged = True
            print(f"[diag] closing aborted early at step {step_i}: object speed={obj_speed:.2f} m/s, "
                  f"displaced {obj_disp:.3f} m from pre-close position")
            break
    print(f"[diag] fingers that locked on contact before full close: {[f for f, v in locked.items() if v]}")

    if diverged:
        print("FAIL: grasp diverged during closing (object got launched by an over-penetrating contact)")
        result["fail_reason"] = "diverged_closing"
        return result

    # ---- diagnostics: what actually happened during closing -------------
    contact_bodies, max_pen = set(), 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        other = b2 if b1 == obj_bid else (b1 if b2 == obj_bid else None)
        if other is not None and other != 0:
            contact_bodies.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other))
            max_pen = min(max_pen, c.dist)
    print(f"[diag] contact bodies at end of close: {sorted(contact_bodies)}")
    print(f"[diag] deepest penetration (negative=inside object): {max_pen:.4f} m")
    track_err = {jn: data.qpos[model.jnt_qposadr[jid]] - close_ctrl[aid] for jn, (jid, aid) in idx.items()}
    worst = sorted(track_err.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
    print("[diag] largest (qpos - close_target) errors, rad (near 0 = actuator swung all the way "
          "to target unopposed = phased through rather than being blocked):")
    for jn, e in worst:
        print(f"    {jn:12s} {e:+.3f}")

    # ---- phase 5: restore gravity, let it load onto the grasp -----------
    model.body_gravcomp[obj_bid] = 0.0
    for _ in range(N_POSTCLOSE_SETTLE):
        step(model, data, viewer)

    close_pos = data.qpos[obj_qadr:obj_qadr + 3].copy()
    print(f"post-close object pos={close_pos.round(3)} (pre-close was {p_world.round(3)}-ish region)")

    # 收攏迴圈裡已經有同一組閾值的即時安全煞車了（見上面 diverged 那段），
    # 這裡是第二道防線，防止 phase 5 的「恢復重力、讓抓取吃到重量」這 300
    # 步和解過程中才發生類似的失控。
    if not np.all(np.isfinite(data.qpos)) or np.linalg.norm(close_pos - obj_pos) > DIVERGE_DISP_M:
        print(f"FAIL: grasp diverged (object displaced >{DIVERGE_DISP_M}m while settling)")
        result["fail_reason"] = "diverged_settling"
        return result

    # ---- phase 6: lift test -----------------------------------------------
    # The object is still resting on the floor at this point. Pushing it
    # sideways/down while the floor (and its friction) is still holding it
    # up proves nothing about the fingers -- an object the hand never
    # actually grasped can "pass" a push test just by sitting there. So
    # before any push test: physically lift the palm (as a real wrist
    # would) and check the object comes with it. From here on the palm is
    # kinematically pinned wherever we put it (qpos rewritten + qvel
    # zeroed every step) -- we're now testing the fingers' grip, not
    # whether an unactuated free-floating wrist also gets knocked around.
    lift_start = data.qpos[palm_qadr:palm_qadr + 7].copy()
    obj_z_before_lift = data.qpos[obj_qadr + 2]
    for i in range(1, N_LIFT_STEPS + 1):
        data.qpos[palm_qadr:palm_qadr + 3] = lift_start[:3] + np.array([0, 0, LIFT_HEIGHT * i / N_LIFT_STEPS])
        data.qpos[palm_qadr + 3:palm_qadr + 7] = lift_start[3:]
        step(model, data, viewer)
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    rise = data.qpos[obj_qadr + 2] - obj_z_before_lift
    print(f"lift test: object rose {rise * 100:.1f}cm (commanded {LIFT_HEIGHT * 100:.0f}cm)")
    if rise < LIFT_SUCCESS_FRACTION * LIFT_HEIGHT:
        print("FAIL: object was not lifted with the hand (never actually grasped)")
        result["fail_reason"] = "not_lifted"
        return result

    palm_hold_pose = data.qpos[palm_qadr:palm_qadr + 7].copy()
    for _ in range(N_POSTLIFT_SETTLE):
        step(model, data, viewer)
        data.qpos[palm_qadr:palm_qadr + 7] = palm_hold_pose
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    ref_pos = data.qpos[obj_qadr:obj_qadr + 3].copy()

    # 拿起測試通過，代表這是一次「確實握住物體」的抓取——把這個當下的姿態
    # 記下來，等一下不管推力測試過不過，都是後續要拿去訓練用的候選資料
    # （doc/plan.md 階段 3 的 object_pose／palm_pose／joint_angles 格式）。
    result["object_pose"] = {
        "pos": ref_pos.tolist(),
        "quat": data.qpos[obj_qadr + 3:obj_qadr + 7].tolist(),
    }
    result["palm_pose"] = {
        "pos": palm_hold_pose[:3].tolist(),
        "quat": palm_hold_pose[3:].tolist(),
    }
    result["joint_angles"] = [float(data.qpos[model.jnt_qposadr[jid]]) for jid, _aid in idx.values()]

    # ---- phase 7: 6-direction perturbation test, object now airborne -----
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    success = True
    for dvec in directions:
        data.xfrc_applied[obj_bid, :3] = dvec * PUSH_FORCE_N
        max_disp = 0.0
        for _ in range(N_PUSH_STEPS):
            step(model, data, viewer)
            data.qpos[palm_qadr:palm_qadr + 7] = palm_hold_pose
            data.qvel[palm_vadr:palm_vadr + 6] = 0.0
            disp = np.linalg.norm(data.qpos[obj_qadr:obj_qadr + 3] - ref_pos)
            max_disp = max(max_disp, disp)
        data.xfrc_applied[obj_bid, :3] = 0.0
        ok = max_disp < SUCCESS_DISP_M
        success &= ok
        print(f"  push {dvec.astype(int)}: max_disp={max_disp:.4f} m  {'OK' if ok else 'FAIL'}")

    print(f"\n{'SUCCESS' if success else 'FAIL'}: grasp {'held' if success else 'did not hold'} under perturbation")
    result["success"] = success
    if not success:
        result["fail_reason"] = "unstable_under_push"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    assets_dir = args.assets.resolve()
    scene_path = build_single_object_scene(assets_dir, args.object)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            run(model, data, args.object, rng, viewer=v)
            print("\n(viewer left open, close the window to exit)")
            while v.is_running():
                time.sleep(0.05)
    else:
        run(model, data, args.object, rng, viewer=None)


if __name__ == "__main__":
    main()
