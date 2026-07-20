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
approaches along `-n`.

Pipeline for one attempt:
  1. Object free-falls onto the floor and settles (as in 07) to get a
     real resting pose; the hand holds a static PREGRASP shape throughout
     so it doesn't disturb anything before the attempt starts.
  2. Sample point `p` + outward normal `n` on the object's *visual* mesh
     (not the convex collision hulls -- see 06_build_objects.py for why),
     transformed into world coordinates via the object's current pose.
  3. Palm is teleported (qpos write, no physics) to `p + n*d`
     (d ~ U(0.08, 0.15)), oriented so its local +Z (== the palm_center
     site direction, i.e. "toward the fingers") points along -n, plus a
     random roll about that axis for wrist variety.
  4. Fingers pre-shape to PREGRASP + noise.
  5. Palm is walked forward in small kinematic steps along -n until any
     hand geom touches the object, or the budget `d` runs out with no
     contact -- most random (p, n, d, roll) samples miss entirely, so
     bailing out here early matters for throughput once this is batched.
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


def step(model, data, viewer=None):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
        time.sleep(0.002)


def run(model, data, obj_name: str, rng: np.random.Generator, viewer=None) -> bool:
    idx = joint_index(model)
    pregrasp_ctrl = hand_pose_ctrl(model, idx, PREGRASP_FRACTION, rng, PREGRASP_NOISE_STD)
    close_ctrl = hand_pose_ctrl(model, idx, CLOSE_FRACTION)

    safe = obj_name.replace("-", "_")
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, safe)
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{safe}_free")
    obj_qadr = model.jnt_qposadr[obj_jid]
    obj_vadr = model.jnt_dofadr[obj_jid]

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

    # ---- phase 2: sample a surface point + outward normal ---------------
    mesh = trimesh.load(Path(__file__).resolve().parents[1]
                         / "assets" / "objects" / obj_name / "visual.stl", force="mesh")
    p_local, face_idx = trimesh.sample.sample_surface(mesh, 1)
    n_local = mesh.face_normals[face_idx[0]]
    p_world = obj_pos + obj_rot @ p_local[0]
    n_world = obj_rot @ n_local
    n_world = n_world / np.linalg.norm(n_world)

    d = rng.uniform(D_MIN, D_MAX)
    roll = rng.uniform(0.0, 2 * np.pi)
    print(f"sampled p={p_world.round(3)} n={n_world.round(3)} d={d:.3f} roll={np.degrees(roll):.0f}deg")

    approach_dir = -n_world  # palm walks toward the surface along -n
    R = rotation_local_z_to(approach_dir, roll)
    quat_xyzw = R.as_quat()
    palm_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    R_mat = R.as_matrix()

    def set_palm(center_pos):
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
        set_palm(p_world + n_world * offset)
        step(model, data, viewer)
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0
        if hand_touches_object(model, data, obj_bid):
            contact_step = i
            break

    if contact_step is None:
        print("FAIL: no contact within approach budget")
        return False
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
    current_ctrl = pregrasp_ctrl.copy()
    model.body_gravcomp[obj_bid] = 1.0
    for step_i in range(N_CLOSE_STEPS):
        frac = min(1.0, (step_i + 1) / N_CLOSE_STEPS)
        for fname, jnames in FINGERS.items():
            if locked[fname]:
                continue
            touched = any(
                (model.geom_bodyid[c.geom1] == obj_bid and model.geom_bodyid[c.geom2] in finger_body_ids[fname])
                or (model.geom_bodyid[c.geom2] == obj_bid and model.geom_bodyid[c.geom1] in finger_body_ids[fname])
                for c in data.contact[:data.ncon]
            )
            if touched:
                locked[fname] = True
                continue
            for jn in jnames:
                _jid, aid = idx[jn]
                current_ctrl[aid] = pregrasp_ctrl[aid] + frac * (close_ctrl[aid] - pregrasp_ctrl[aid])
        data.ctrl[:] = current_ctrl
        step(model, data, viewer)
        if all(locked.values()):
            break
    print(f"[diag] fingers that locked on contact before full close: {[f for f, v in locked.items() if v]}")

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

    # A bad grasp can drive many hand geoms into the object at once and
    # blow up the contact solver -- qpos stays finite but the object ends
    # up meters away. Bail before the lift/push tests waste time on an
    # already-obviously-failed candidate.
    if not np.all(np.isfinite(data.qpos)) or np.linalg.norm(close_pos - obj_pos) > 0.5:
        print("FAIL: grasp diverged (object displaced >0.5m during closing)")
        return False

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
        return False

    palm_hold_pose = data.qpos[palm_qadr:palm_qadr + 7].copy()
    for _ in range(N_POSTLIFT_SETTLE):
        step(model, data, viewer)
        data.qpos[palm_qadr:palm_qadr + 7] = palm_hold_pose
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    ref_pos = data.qpos[obj_qadr:obj_qadr + 3].copy()

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
    return success


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
