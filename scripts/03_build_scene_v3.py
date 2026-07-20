#!/usr/bin/env python3
"""
03_build_scene_v3.py
--------------------
Why v3 exists
=============
In the DG-5F URDF the chain mount -> base -> palm is joined by *fixed*
joints, and the root link has no joint connecting it to the world. When
MuJoCo imports that URDF it therefore:

  * welds mount/base/palm into the world, emitting their geoms as loose
    <geom> elements directly inside <worldbody>, and
  * promotes the five finger roots (ll_dg_N_1) to top-level bodies.

So there is no palm body to attach a freejoint to, and the hand is not a
single kinematic tree. v3 rebuilds it: a new body "palm" is created, the
loose worldbody geoms and the five finger bodies are moved inside it,
and the freejoint goes on that new body.

Usage:
    python 03_build_scene_v3.py --assets ../assets
    python -m mujoco.viewer --mjcf=../assets/dg5f_scene.xml
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

FINGER_JOINTS = [f"lj_dg_{f}_{k}" for f in range(1, 6) for k in range(1, 5)]
FINGER_ROOTS = [f"ll_dg_{f}_1" for f in range(1, 6)]

# Fixed offset from the URDF: mount -> base is +0.004, base -> palm is
# +0.0698. Finger bodies are already emitted in mount coordinates, so the
# new palm body sits at the origin and nothing needs re-offsetting.
PALM_HEIGHT = 0.0738


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--spawn-z", type=float, default=0.30)
    args = ap.parse_args()

    assets = args.assets.resolve()
    urdf = assets / "dg5f_mj.urdf"
    raw_xml = assets / "_dg5f_raw.xml"
    scene_xml = assets / "dg5f_scene.xml"

    model = mujoco.MjModel.from_xml_path(str(urdf))
    mujoco.mj_saveLastXML(str(raw_xml), model)
    print(f"raw MJCF -> {raw_xml}")

    tree = ET.parse(raw_xml)
    root = tree.getroot()

    def sub(parent, tag, **kw):
        el = parent.find(tag)
        if el is None:
            el = ET.SubElement(parent, tag)
        el.attrib.update({k: str(v) for k, v in kw.items()})
        return el

    comp = sub(root, "compiler", angle="radian", autolimits="true")
    comp.attrib.pop("meshdir", None)

    sub(root, "option",
        timestep=0.002,
        integrator="implicitfast",
        cone="elliptic",
        impratio=10,
        noslip_iterations=5)

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(0, default)
    hand_def = ET.SubElement(default, "default", {"class": "hand"})
    ET.SubElement(hand_def, "geom", {
        "condim": "4",
        "friction": "1.0 0.01 0.001",
        "solref": "0.005 1",
        "solimp": "0.9 0.98 0.001",
        "margin": "0.0005",
        "rgba": "0.75 0.77 0.80 1",
    })
    ET.SubElement(hand_def, "joint", {"damping": "0.1", "armature": "0.005"})

    worldbody = root.find("worldbody")

    # ---- collect the welded palm geoms and the finger bodies -----------
    loose_geoms = [g for g in worldbody.findall("geom")]
    finger_bodies = [b for b in worldbody.findall("body")]

    print(f"found {len(loose_geoms)} welded palm geoms, "
          f"{len(finger_bodies)} finger bodies")
    got = [b.get("name") for b in finger_bodies]
    missing = [n for n in FINGER_ROOTS if n not in got]
    if missing:
        raise RuntimeError(f"missing expected finger roots: {missing}\ngot: {got}")

    # ---- build the new palm body ---------------------------------------
    palm = ET.Element("body", {"name": "palm", "pos": f"0 0 {args.spawn_z}"})
    ET.SubElement(palm, "freejoint", {"name": "hand_free"})

    # An explicit inertial is required: the welded palm geoms carry no
    # inertia of their own once they are moved into a new body.
    ET.SubElement(palm, "inertial", {
        "pos": "-0.006 -0.001 0.040",
        "mass": "0.85",                      # mount 0.05 + base 0.45 + palm 0.35
        "diaginertia": "1.06e-3 9.5e-4 9.3e-4",
    })

    for g in loose_geoms:
        worldbody.remove(g)
        palm.append(g)
    for b in finger_bodies:
        worldbody.remove(b)
        palm.append(b)

    worldbody.insert(0, palm)
    print(f"palm body created: {len(loose_geoms)} geoms + "
          f"{len(finger_bodies)} finger children, freejoint attached")

    for g in palm.iter("geom"):
        g.set("class", "hand")
    for j in palm.iter("joint"):
        j.set("class", "hand")

    # a site at the palm centre, used later as the grasp reference frame
    ET.SubElement(palm, "site", {
        "name": "palm_center",
        "pos": f"0 0 {PALM_HEIGHT}",
        "size": "0.005",
        "rgba": "1 0 0 0.6",
    })

    # ---- world furniture ------------------------------------------------
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "1 1 0.05",
        "pos": "0 0 0", "rgba": "0.4 0.42 0.45 1",
        "condim": "3", "friction": "1.0 0.005 0.0001"})

    # ---- actuators ------------------------------------------------------
    for old in root.findall("actuator"):
        root.remove(old)
    act = ET.SubElement(root, "actuator")

    names_in_model = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    }
    n_act = 0
    for jn in FINGER_JOINTS:
        if jn not in names_in_model:
            print(f"  ! joint {jn} missing, skipped")
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo, hi = model.jnt_range[jid]
        ET.SubElement(act, "position", {
            "name": f"act_{jn}", "joint": jn,
            "ctrlrange": f"{lo} {hi}",
            "kp": "3.0", "kv": "0.2",
            "forcerange": "-7.5 7.5",
        })
        n_act += 1
    print(f"{n_act} position actuators added")

    ET.indent(tree, space="  ")
    tree.write(scene_xml, encoding="utf-8", xml_declaration=True)
    print(f"scene -> {scene_xml}")

    # ---- sanity check ---------------------------------------------------
    m2 = mujoco.MjModel.from_xml_path(str(scene_xml))
    d2 = mujoco.MjData(m2)
    m2.body_gravcomp[:] = 1.0
    for _ in range(500):
        mujoco.mj_step(m2, d2)

    pid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "palm")
    drift = np.linalg.norm(d2.xpos[pid] - np.array([0, 0, args.spawn_z]))

    print("\n=== SANITY CHECK ===")
    print(f"nq={m2.nq}  nv={m2.nv}  nu={m2.nu}  nbody={m2.nbody}  ngeom={m2.ngeom}")
    print("expected: nq=27  nv=26  nu=20  nbody=22 (world+palm+20 finger links)")
    print(f"finite after 500 steps : {np.all(np.isfinite(d2.qpos))}")
    print(f"palm drift (gravcomp on): {drift:.5f} m   (should be ~0)")

    ok = (m2.nq == 27 and m2.nu == 20
          and np.all(np.isfinite(d2.qpos)) and drift < 1e-3)
    print(f"\nSTAGE 1 BUILD {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()