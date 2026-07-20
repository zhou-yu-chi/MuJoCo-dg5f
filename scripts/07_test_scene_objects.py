#!/usr/bin/env python3
"""
07_test_scene_objects.py
-------------------------
Stage-2 acceptance test: load the hand + all 5 YCB objects together
(assets/dg5f_scene_objects.xml) and confirm nothing is broken.

The hand's palm is pinned in place the same way 04_test_close.py does it
(see that script's docstring for why an unanchored freejoint is the wrong
thing to test against). The 5 objects are spawned in a row well off to
the side of the hand -- object.xml gives every object the same default
spawn pose, so distinct, non-overlapping positions are set here at
runtime rather than baked into each fragment.

Objects are left under real gravity (only the hand gets gravcomp) and
allowed to free-fall ~15cm onto the floor, so this also doubles as a
sanity check that the VHACD collision hulls from 06_build_objects.py
don't blow up on contact.

Usage:
    python 07_test_scene_objects.py --scene ../assets/dg5f_scene_objects.xml
    python 07_test_scene_objects.py --scene ../assets/dg5f_scene_objects.xml --viewer
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "011_banana",
    "025_mug",
    "065_a_cups",  # sanitized: "-" -> "_", matches 06_build_objects.py
]

DROP_X = 0.5       # spawn row is offset well clear of the hand at x=0
DROP_Z = 0.15
SETTLE_STEPS = 1500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    # gravcomp only the hand (palm + finger links); objects must actually fall.
    obj_body_ids = set()
    for name in OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"body '{name}' not found -- check OBJECTS list / object.xml names")
        obj_body_ids.add(bid)
    for bid in range(model.nbody):
        if bid not in obj_body_ids:
            model.body_gravcomp[bid] = 1.0

    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
    palm_qadr = model.jnt_qposadr[free_jid]
    palm_vadr = model.jnt_dofadr[free_jid]
    palm_pin = data.qpos[palm_qadr:palm_qadr + 7].copy()

    # place objects in a row, spaced 0.25m apart in y, off to the side in x
    obj_info = []  # (name, qadr, vadr)
    ys = np.linspace(-0.5, 0.5, len(OBJECTS))
    for name, y in zip(OBJECTS, ys):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]
        data.qpos[qadr:qadr + 3] = [DROP_X, y, DROP_Z]
        data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        obj_info.append((name, qadr, vadr))

    def pin_palm():
        data.qpos[palm_qadr:palm_qadr + 7] = palm_pin
        data.qvel[palm_vadr:palm_vadr + 6] = 0.0

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                pin_palm()
                v.sync()
                time.sleep(0.002)
        return

    # ---------------- headless acceptance check ----------------
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        pin_palm()

    print("=== object resting state after free-fall ===")
    ok = True
    for name, qadr, vadr in obj_info:
        pos = data.qpos[qadr:qadr + 3]
        speed = np.linalg.norm(data.qvel[vadr:vadr + 3])
        finite = np.all(np.isfinite(data.qpos[qadr:qadr + 7]))
        settled = finite and speed < 0.01 and -0.05 < pos[2] < 0.30
        ok &= settled
        flag = "OK " if settled else "BAD"
        print(f"  {flag} {name:20s} pos={pos.round(3)} speed={speed:.4f} finite={finite}")

    print(f"\nSTAGE 2 LOAD CHECK {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
