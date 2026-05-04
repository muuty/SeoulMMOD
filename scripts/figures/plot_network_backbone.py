from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def parquet_glob(data_root: Path) -> str:
    return str(data_root / "raw" / "od_flow_*.parquet")


def year_filter(year: int) -> str:
    return f"date >= DATE '{year}-01-01' AND date < DATE '{year + 1}-01-01'"


def disparity_filter(matrix: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    mask = np.zeros(matrix.shape, dtype=bool)
    for i, row in enumerate(matrix):
        row_sum = row.sum()
        if row_sum == 0:
            continue
        degree = np.count_nonzero(row)
        if degree <= 1:
            mask[i, row > 0] = True
            continue
        pvals = np.power(1.0 - row / row_sum, degree - 1)
        mask[i, (pvals < alpha) & (row > 0)] = True
    return mask | mask.T


def to_graph(matrix: np.ndarray) -> nx.Graph:
    sym = (matrix + matrix.T) / 2.0
    graph = nx.Graph()
    graph.add_nodes_from(range(sym.shape[0]))
    rows, cols = np.nonzero(sym)
    for i, j in zip(rows, cols):
        if i < j:
            graph.add_edge(int(i), int(j), weight=float(sym[i, j]))
    return graph


def load_geometry(data_root: Path) -> tuple[gpd.GeoDataFrame, list[str], dict[int, tuple[float, float]]]:
    gdf = gpd.read_file(data_root / "subdistrict_boundaries.geojson")
    gdf["subdistrict_cd"] = gdf["subdistrict_cd"].astype(str)
    gdf = gdf.sort_values("subdistrict_cd").reset_index(drop=True)
    projected = gdf.to_crs(5179)
    centroids = {
        i: (float(geom.centroid.x), float(geom.centroid.y))
        for i, geom in enumerate(projected.geometry)
    }
    return projected, gdf["subdistrict_cd"].tolist(), centroids


def load_matrix(
    con: duckdb.DuckDBPyConnection,
    data_glob: str,
    codes: list[str],
    mode: int,
    year: int,
) -> np.ndarray:
    code_df = pd.DataFrame({"subdistrict_cd": codes, "idx": range(len(codes))})
    con.register("codes", code_df)
    df = con.execute(
        f"""
        SELECT o.idx AS origin_idx, d.idx AS dest_idx, SUM(flow) AS weight
        FROM read_parquet('{data_glob}', union_by_name=true) t
        JOIN codes o ON t.origin = o.subdistrict_cd
        JOIN codes d ON t.dest = d.subdistrict_cd
        WHERE {year_filter(year)} AND mode = {mode}
        GROUP BY o.idx, d.idx
        """
    ).df()
    matrix = np.zeros((len(codes), len(codes)), dtype=np.float64)
    matrix[df["origin_idx"].to_numpy(), df["dest_idx"].to_numpy()] = df["weight"].to_numpy(dtype=float)
    return matrix


def louvain_partition(graph: nx.Graph) -> dict[int, int]:
    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    return {int(node): label for label, nodes in enumerate(communities) for node in nodes}


def plot_backbone(
    gdf: gpd.GeoDataFrame,
    codes: list[str],
    centroids: dict[int, tuple[float, float]],
    matrix: np.ndarray,
    output_dir: Path,
    mode: int,
) -> None:
    backbone = np.where(disparity_filter(matrix), matrix, 0.0)
    graph = to_graph(backbone)
    partition = louvain_partition(graph)
    clusters = len(set(partition.values()))
    cluster_df = pd.DataFrame(
        {
            "subdistrict_cd": codes,
            "cluster": [partition.get(i, -1) for i in range(len(codes))],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.merge(cluster_df, on="subdistrict_cd", how="left").plot(
        column="cluster",
        cmap=plt.get_cmap("tab20", max(clusters, 10)),
        ax=ax,
        edgecolor="white",
        linewidth=0.18,
        categorical=True,
        legend=False,
        alpha=0.5,
    )

    sym = (backbone + backbone.T) / 2.0
    rows, cols = np.nonzero(sym)
    max_weight = float(sym[rows, cols].max()) if len(rows) else 1.0
    alpha_base, alpha_span = ((0.2, 0.25) if mode == 8 else (0.3, 0.55))
    line_base = 0.3 if mode == 8 else 0.8
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        strength = float(sym[i, j]) / max_weight
        x = [centroids[int(i)][0], centroids[int(j)][0]]
        y = [centroids[int(i)][1], centroids[int(j)][1]]
        ax.plot(x, y, color="black", alpha=alpha_base + strength * alpha_span, linewidth=line_base + strength * 2.2)

    ax.axis("off")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "fig_07_network_car.png" if mode == 8 else "fig_08_network_walk.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    con = duckdb.connect()
    gdf, codes, centroids = load_geometry(args.data_root)
    data_glob = parquet_glob(args.data_root)
    for mode in [8, 7]:
        matrix = load_matrix(con, data_glob, codes, mode, args.year)
        plot_backbone(gdf, codes, centroids, matrix, args.output_dir, mode)


if __name__ == "__main__":
    main()
