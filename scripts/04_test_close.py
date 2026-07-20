#!/usr/bin/env python3
"""
04_test_close.py  (v2)
----------------------
Stage-1 acceptance test for the DG-5F MuJoCo scene.

Fixes over v1:
  * `import mujoco.viewer` was inside main(), which made the name
    `mujoco` local to the entire function and shadowed the module-level
    import -> UnboundLocalError on the first use. Moved to module level.
  * gravity compensation is now applied before MjData is created and is
    reported, so the hand reliably hovers during the test.
  * per-joint signed travel is printed, not just the magnitude, so a
    finger curling the *wrong way* is visible instead of silently
    counting as travel.

Usage:
    python 04_test_close.py --scene ../assets/dg5f_scene.xml
    python 04_test_close.py --scene ../assets/dg5f_scene.xml --viewer
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer          # must be at module level, see docstring
import numpy as np

FINGERS = {
    "thumb":  [f"lj_dg_1_{k}" for k in range(1, 5)],
    "index":  [f"lj_dg_2_{k}" for k in range(1, 5)],
    "middle": [f"lj_dg_3_{k}" for k in range(1, 5)],
    "ring":   [f"lj_dg_4_{k}" for k in range(1, 5)],
    "pinky":  [f"lj_dg_5_{k}" for k in range(1, 5)],
}

CLOSE_FRACTION = 0.60

# viewer demo timing: DG-5F is direct-drive (20 independently actuated
# joints, no tendon coupling -- confirmed against Tesollo's spec), but the
# real hand still servos smoothly over ~1-2s per open/close stroke. The
# ctrl target must therefore be ramped, not stepped, or the low-inertia
# sim (armature=0.005) snaps to target in ~0.1s and looks like a twitch.
CYCLE_S = 6.0          # full open -> close -> open period
MOVE_S = 2.0           # ramp duration for each open/close stroke


def ease(x: float) -> float:
    """Smoothstep in [0, 1]: zero velocity at both ends of the ramp."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_index(model):
    """Map joint name -> (joint id, qpos address, actuator id)."""
    idx = {}
    for jnames in FINGERS.values():
        for jn in jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jn}")
            if jid < 0 or aid < 0:
                raise RuntimeError(f"{jn}: joint id {jid}, actuator id {aid}")
            idx[jn] = (jid, model.jnt_qposadr[jid], aid)
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


def make_poses(model, idx):
    """OPEN and CLOSED ctrl vectors derived from joint limits.

    Spread/rotation joints are held neutral. True flexion joints are
    swept from open toward the verified closing direction (see
    flexion_open_close).
    """
    open_ctrl = np.zeros(model.nu)
    close_ctrl = np.zeros(model.nu)
    for jn, (jid, _adr, aid) in idx.items():
        lo, hi = model.jnt_range[jid]
        bounds = flexion_open_close(jn, lo, hi)
        if bounds is None:
            open_ctrl[aid] = close_ctrl[aid] = float(np.clip(0.0, lo, hi))
        else:
            near, far = bounds
            open_ctrl[aid] = near
            close_ctrl[aid] = near + CLOSE_FRACTION * (far - near)
    return open_ctrl, close_ctrl


def settle(model, data, ctrl, n=1500, palm_pin=None):
    """Step the sim. If palm_pin=(qadr, vadr, qpos7) is given, the palm
    freejoint is reset to that pose every step -- see note in main() for
    why an unpinned free-floating palm makes this test meaningless."""
    data.ctrl[:] = ctrl
    for _ in range(n):
        mujoco.mj_step(model, data)
        if palm_pin is not None:
            qadr, vadr, qpos7 = palm_pin
            data.qpos[qadr:qadr + 7] = qpos7
            data.qvel[vadr:vadr + 6] = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    # cancel gravity so the free-floating hand hovers during the test
    model.body_gravcomp[:] = 1.0
    data = mujoco.MjData(model)

    idx = build_index(model)
    open_ctrl, close_ctrl = make_poses(model, idx)

    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    # The palm rides on a freejoint with nothing else anchoring it. When the
    # finger actuators apply torque to close the fingers, Newton's third law
    # sends an equal-and-opposite reaction into the palm -- a free-floating
    # body has no restoring force to resist that, so it drifts/tips even
    # with zero bugs in the finger control. That's real physics, not what
    # this test is trying to measure. Stage 3 will set palm pose externally
    # anyway, so pin it here to isolate finger-tracking accuracy.
    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
    palm_qadr = model.jnt_qposadr[free_jid]
    palm_vadr = model.jnt_dofadr[free_jid]
    palm_pin = (palm_qadr, palm_vadr, data.qpos[palm_qadr:palm_qadr + 7].copy())

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            t0 = time.time()
            half = CYCLE_S / 2.0
            while v.is_running():
                t = (time.time() - t0) % CYCLE_S
                if t < half:
                    frac = ease(min(t / MOVE_S, 1.0))
                    target = open_ctrl + frac * (close_ctrl - open_ctrl)
                else:
                    frac = ease(min((t - half) / MOVE_S, 1.0))
                    target = close_ctrl + frac * (open_ctrl - close_ctrl)
                data.ctrl[:] = target
                mujoco.mj_step(model, data)
                data.qpos[palm_qadr:palm_qadr + 7] = palm_pin[2]
                data.qvel[palm_vadr:palm_vadr + 6] = 0.0
                v.sync()
                time.sleep(0.002)
        return

    # ---------------- headless acceptance check ----------------
    settle(model, data, open_ctrl, palm_pin=palm_pin)
    q_open = data.qpos.copy()
    palm_open = data.xpos[pid].copy()

    settle(model, data, close_ctrl, palm_pin=palm_pin)
    q_close = data.qpos.copy()
    palm_close = data.xpos[pid].copy()

    print(f"=== contacts at closed pose: {data.ncon} ===")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"  {g1!s:>20} <-> {g2!s:<20}  dist={c.dist:+.4f}")

    print("\n=== per-joint signed travel (rad) ===")
    ok = True
    for fname, jnames in FINGERS.items():
        parts, total = [], 0.0
        for jn in jnames:
            _jid, adr, _aid = idx[jn]
            d = q_close[adr] - q_open[adr]
            parts.append(f"{d:+.2f}")
            total += abs(d)
        flag = "OK " if total > 1.0 else "BAD"
        if total <= 1.0:
            ok = False
        print(f"  {flag} {fname:7s} |total|={total:5.2f}   " + "  ".join(parts))

    # tracking error, actuator-space (skip the 7 freejoint qpos entries)
    errs = []
    for jn, (_jid, adr, aid) in idx.items():
        errs.append(abs(q_close[adr] - close_ctrl[aid]))
    err = max(errs)

    palm_drift = np.linalg.norm(palm_close - palm_open)

    print(f"\nmax tracking error at closed pose : {err:.4f} rad")
    print(f"palm drift open->closed           : {palm_drift:.5f} m")

    passed = ok and err < 0.15 and palm_drift < 0.02
    print(f"\nSTAGE 1 {'PASSED' if passed else 'FAILED'}")
    if not passed:
        if err >= 0.15:
            print("  - actuators not reaching target: raise kp, or self-collision")
        if palm_drift >= 0.02:
            print("  - palm moved: fingers are pushing on each other or the floor")


if __name__ == "__main__":
    main()