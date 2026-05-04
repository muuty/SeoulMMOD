from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np


SPECS = {
    "district": {
        "dataset": "SeoulMMOD_District_2024",
        "boundary": "kaggle_seoulmmod/district_boundaries.geojson",
        "code_col": "district_cd",
    },
    "subdistrict": {
        "dataset": "SeoulMMOD_Subdistrict_2024",
        "boundary": "kaggle_seoulmmod/subdistrict_boundaries.geojson",
        "code_col": "subdistrict_cd",
    },
}


def node_order(dataset_dir: Path) -> list[str]:
    with open(dataset_dir / "od_pairs.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)["od_pairs"]
    origins = []
    for origin, _ in pairs:
        if origin not in origins:
            origins.append(origin)
    dests = []
    for _, dest in pairs:
        if dest not in dests:
            dests.append(dest)
    if origins != dests:
        raise ValueError(f"origin/destination node order mismatch in {dataset_dir}")
    if len(origins) * len(dests) != len(pairs):
        raise ValueError(f"OD pairs are not a complete square matrix in {dataset_dir}")
    return origins


def build_adjacency(boundary_path: Path, code_col: str, codes: list[str]) -> np.ndarray:
    gdf = gpd.read_file(boundary_path)
    gdf[code_col] = gdf[code_col].astype(str)
    gdf = gdf.drop_duplicates(code_col).set_index(code_col)
    missing = sorted(set(codes) - set(gdf.index))
    extra = sorted(set(gdf.index) - set(codes))
    if missing or extra:
        raise ValueError(
            f"boundary/dataset code mismatch for {boundary_path}: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    gdf = gdf.loc[codes]
    geoms = gdf.geometry
    adj = np.zeros((len(codes), len(codes)), dtype=np.int8)
    for i, geom in enumerate(geoms):
        adj[i] = geoms.touches(geom).to_numpy(dtype=np.int8)
    np.fill_diagonal(adj, 0)
    if not np.array_equal(adj, adj.T):
        adj = np.maximum(adj, adj.T).astype(np.int8)
    return adj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--level", choices=["district", "subdistrict", "all"], default="all")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    levels = ["district", "subdistrict"] if args.level == "all" else [args.level]
    for level in levels:
        spec = SPECS[level]
        dataset_dir = root / "datasets" / spec["dataset"]
        boundary_path = root.parent / spec["boundary"]
        codes = node_order(dataset_dir)
        adj = build_adjacency(boundary_path, spec["code_col"], codes)
        np.save(dataset_dir / "adjacency_matrix.npy", adj)
        np.save(dataset_dir / "adj_matrix.npy", adj)
        np.savetxt(dataset_dir / "adjacency_matrix.csv", adj, fmt="%d")
        np.savetxt(dataset_dir / "adj_matrix.csv", adj, fmt="%d")
        print(
            f"{spec['dataset']}: shape={adj.shape}, "
            f"edges={int(adj.sum() // 2)}, isolated={int((adj.sum(axis=1) == 0).sum())}"
        )


if __name__ == "__main__":
    main()
