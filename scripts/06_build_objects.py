#!/usr/bin/env python3
"""
06_build_objects.py
--------------------
Stage-2: turn the raw YCB `nontextured.stl` meshes (already downloaded into
assets/objects/<name>/google_16k/) into MJCF fragments that can be
<include>d next to the hand.

Why visual and collision meshes are split, unlike the hand in 01/03:
  * MuJoCo collision only works on convex geoms, so every object needs a
    VHACD convex decomposition for physics -- same reason as the hand.
  * But for objects (unlike the hand) later stages render synthetic depth
    images (plan.md stage 4) off these meshes, and a mug's handle hole or
    a banana's curve would visibly distort into a convex blob if the
    render used the collision hulls. So objects keep the full-resolution
    source mesh for visual (non-colliding) and only use the convex hulls
    for physics.

Mesh origin: the raw YCB scan origin is arbitrary, not the object's
center of mass. Both visual and collision meshes are re-centered on the
source mesh's center of mass before export, so the body frame origin
(where the freejoint and any later grasp-pose math lives) sits at the
object's actual center, not some corner of the scan volume.

Mass/inertia: left to MuJoCo's per-geom `density` (default 1000, i.e.
water) rather than measured YCB weights -- fine for "does it load and
collide sensibly" in stage 2, revisit with the published YCB masses if
stage 3 grasp stability turns out to be sensitive to it.

Usage:
    python 06_build_objects.py --assets ../assets
"""

import argparse
from pathlib import Path

import trimesh

OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "011_banana",
    "025_mug",
    "065-a_cups",
]

MAX_HULLS = 32


def decompose(mesh: trimesh.Trimesh, dst_dir: Path, stem: str) -> list[str]:
    """VHACD-decompose an already-centered mesh. Returns written filenames."""
    if mesh.is_convex:
        out = dst_dir / f"{stem}.stl"
        mesh.export(out)
        return [out.name]

    try:
        parts = mesh.convex_decomposition(maxConvexHulls=MAX_HULLS, resolution=100_000)
    except Exception as e:  # noqa: BLE001
        print(f"    ! VHACD failed ({e}); falling back to convex hull")
        parts = [mesh.convex_hull]

    if isinstance(parts, trimesh.Trimesh):
        parts = [parts]

    names = []
    for i, p in enumerate(parts):
        if p.volume < 1e-9:  # discard slivers
            continue
        out = dst_dir / f"{stem}_{i:02d}.stl"
        p.export(out)
        names.append(out.name)

    if not names:  # everything got discarded
        out = dst_dir / f"{stem}.stl"
        mesh.convex_hull.export(out)
        names = [out.name]
    return names


def build_one(obj_dir: Path, name: str) -> str:
    """Process one object dir in-place, return an MJCF fragment string."""
    src = obj_dir / "google_16k" / "nontextured.stl"
    mesh = trimesh.load(src, force="mesh")

    com = mesh.center_mass if mesh.is_watertight else mesh.centroid
    mesh.apply_translation(-com)

    visual_path = obj_dir / "visual.stl"
    mesh.export(visual_path)

    col_dir = obj_dir / "collision"
    col_dir.mkdir(exist_ok=True)
    hull_files = decompose(mesh, col_dir, name)

    print(f"  {name:22s} watertight={mesh.is_watertight!s:5s} "
          f"-> {len(hull_files)} collision hull(s)")

    safe = name.replace("-", "_")
    mesh_tags = [f'    <mesh name="{safe}_visual" file="objects/{name}/visual.stl" />']
    geom_tags = [
        f'      <geom type="mesh" mesh="{safe}_visual" contype="0" conaffinity="0" '
        f'group="2" rgba="0.8 0.8 0.75 1" />'
    ]
    for i, fn in enumerate(hull_files):
        mesh_name = f"{safe}_col_{i:02d}"
        mesh_tags.append(f'    <mesh name="{mesh_name}" file="objects/{name}/collision/{fn}" />')
        geom_tags.append(
            f'      <geom type="mesh" mesh="{mesh_name}" group="3" condim="4" '
            f'friction="0.8 0.01 0.001" density="1000" '
            # Match the hand's "hand" default class (dg5f_scene.xml) exactly.
            # Left at MuJoCo's engine default (solref="0.02 1"), a hand-object
            # contact gets the *average* of the two geoms' stiffness (equal
            # priority => solmix averaging) -- 2.5x softer than the hand alone,
            # which is what let fingers visibly sink ~2cm into a closed grasp.
            f'solref="0.005 1" solimp="0.9 0.98 0.001" />'
        )

    fragment = (
        "<mujoco>\n"
        "  <asset>\n" + "\n".join(mesh_tags) + "\n  </asset>\n"
        "  <worldbody>\n"
        f'    <body name="{safe}" pos="0 0 0.5">\n'
        f'      <freejoint name="{safe}_free" />\n'
        + "\n".join(geom_tags) + "\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    return fragment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, type=Path)
    args = ap.parse_args()

    objects_root = args.assets.resolve() / "objects"

    print("=== building object MJCF fragments ===")
    for name in OBJECTS:
        obj_dir = objects_root / name
        if not obj_dir.exists():
            raise FileNotFoundError(f"missing {obj_dir}, run the download step first")
        fragment = build_one(obj_dir, name)
        out_xml = obj_dir / "object.xml"
        out_xml.write_text(fragment)
        print(f"    -> {out_xml}")


if __name__ == "__main__":
    main()
