#!/usr/bin/env python3
"""
01_convert_urdf.py
------------------
Convert the Tesollo DG-5F URDF into a MuJoCo-loadable form.

What it does:
  1. Rewrite package:// mesh paths to relative paths.
  2. Drop the .dae visual meshes (MuJoCo can't read COLLADA) and reuse
     the collision STLs for visuals.
  3. Run VHACD convex decomposition on every collision STL, because
     MuJoCo only does convex-convex collision. Non-convex fingertips
     would otherwise grab objects through the mesh.
  4. Write assets/dg5f_mj.urdf + assets/meshes/*.stl

Usage:
    python 01_convert_urdf.py \
        --urdf /home/rex/桌面/dg5f/delto_m_ros2/dg_description/urdf/dg5f_left.urdf \
        --pkg  /home/rex/桌面/dg5f/delto_m_ros2/dg_description \
        --out  ../assets
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def resolve_pkg(filename: str, pkg_root: Path) -> Path:
    """package://dg_description/meshes/... -> <pkg_root>/meshes/..."""
    assert filename.startswith("package://"), filename
    rest = filename[len("package://"):]
    _, subpath = rest.split("/", 1)
    return pkg_root / subpath


def decompose(src: Path, dst_dir: Path, max_hulls: int = 8) -> list[str]:
    """VHACD-decompose one STL. Returns list of written filenames."""
    mesh = trimesh.load(src, force="mesh")

    # Already convex within tolerance -> no need to split.
    if mesh.is_convex:
        out = dst_dir / f"{src.stem}.stl"
        mesh.export(out)
        return [out.name]

    try:
        parts = mesh.convex_decomposition(maxConvexHulls=max_hulls, resolution=100_000)
    except Exception as e:  # noqa: BLE001
        print(f"    ! VHACD failed on {src.name} ({e}); falling back to convex hull")
        parts = [mesh.convex_hull]

    if isinstance(parts, trimesh.Trimesh):
        parts = [parts]

    names = []
    for i, p in enumerate(parts):
        if p.volume < 1e-9:  # discard slivers
            continue
        out = dst_dir / f"{src.stem}_{i:02d}.stl"
        p.export(out)
        names.append(out.name)

    if not names:  # everything got discarded
        out = dst_dir / f"{src.stem}.stl"
        mesh.convex_hull.export(out)
        names = [out.name]
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=Path)
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-hulls", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out.resolve()
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.urdf)
    root = tree.getroot()

    cache: dict[str, list[str]] = {}

    for link in root.findall("link"):
        name = link.get("name")

        # --- collision: decompose, may become several <collision> blocks ---
        cols = link.findall("collision")
        for col in cols:
            link.remove(col)

        new_col_files: list[str] = []
        for col in cols:
            m = col.find(".//mesh")
            if m is None:
                link.append(col)  # primitive geometry, keep as-is
                continue
            fn = m.get("filename")
            if fn not in cache:
                src = resolve_pkg(fn, args.pkg)
                if not src.exists():
                    raise FileNotFoundError(src)
                print(f"  decomposing {src.name} ...")
                cache[fn] = decompose(src, mesh_dir, args.max_hulls)
                print(f"    -> {len(cache[fn])} hull(s)")
            new_col_files.extend(cache[fn])

        for f in new_col_files:
            c = ET.SubElement(link, "collision")
            g = ET.SubElement(c, "geometry")
            ET.SubElement(g, "mesh", filename=f"meshes/{f}")

        # --- visual: replace .dae with the same convex hulls ---
        for vis in link.findall("visual"):
            link.remove(vis)
        for f in new_col_files:
            v = ET.SubElement(link, "visual")
            g = ET.SubElement(v, "geometry")
            ET.SubElement(g, "mesh", filename=f"meshes/{f}")

        if not new_col_files:
            print(f"  ! link '{name}' has no collision mesh")

    out_urdf = out_dir / "dg5f_mj.urdf"
    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    print(f"\nwrote {out_urdf}")
    print(f"meshes in {mesh_dir}  ({len(list(mesh_dir.glob('*.stl')))} files)")


if __name__ == "__main__":
    main()
