from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


MODE_NAMES = {
    4: "Metro Bus",
    5: "Local Bus",
    6: "Subway",
    7: "Walk",
    8: "Car",
    9: "Other",
}
MODE_COLORS = {
    "Metro Bus": "#FF9F40",
    "Local Bus": "#FFCF40",
    "Subway": "#E73C3C",
    "Walk": "#4CAF50",
    "Car": "#2196F3",
    "Other": "#9C27B0",
}
MODE_ORDER = [4, 5, 6, 7, 8, 9]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    return parser.parse_args()


def parquet_glob(data_root: Path) -> str:
    return str(data_root / "raw" / "od_flow_*.parquet")


def year_filter(year: int, column: str = "date") -> str:
    return f"{column} >= DATE '{year}-01-01' AND {column} < DATE '{year + 1}-01-01'"


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )
    sns.set_style("whitegrid")


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flow_distribution(
    con: duckdb.DuckDBPyConnection,
    data_glob: str,
    output_dir: Path,
    sample_size: int,
) -> None:
    df = con.execute(
        f"""
        SELECT flow
        FROM read_parquet('{data_glob}', union_by_name=true)
        WHERE flow > 0
        USING SAMPLE {sample_size}
        """
    ).df()
    flow = df["flow"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(flow, bins=np.logspace(np.log10(flow.min()), np.log10(flow.max()), 80), color="steelblue", alpha=0.8)
    ax.axvline(np.median(flow), color="red", linestyle="--", linewidth=1.5, label=f"Median={np.median(flow):.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flow")
    ax.set_ylabel("Count (Log)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "fig_01a_flow_hist.png")


def plot_net_inflow(con: duckdb.DuckDBPyConnection, data_root: Path, data_glob: str, output_dir: Path, year: int) -> None:
    gdf = gpd.read_file(data_root / "subdistrict_boundaries.geojson")
    gdf["subdistrict_cd"] = gdf["subdistrict_cd"].astype(str)
    df = con.execute(
        f"""
        WITH rush AS (
            SELECT origin, dest, flow
            FROM read_parquet('{data_glob}', union_by_name=true)
            WHERE {year_filter(year)} AND st_hour BETWEEN 7 AND 9
        ),
        outflow AS (
            SELECT origin AS subdistrict_cd, SUM(flow) AS outflow
            FROM rush
            GROUP BY origin
        ),
        inflow AS (
            SELECT dest AS subdistrict_cd, SUM(flow) AS inflow
            FROM rush
            GROUP BY dest
        )
        SELECT
            COALESCE(i.subdistrict_cd, o.subdistrict_cd) AS subdistrict_cd,
            COALESCE(inflow, 0) - COALESCE(outflow, 0) AS net_inflow
        FROM inflow i
        FULL OUTER JOIN outflow o USING (subdistrict_cd)
        """
    ).df()
    df["subdistrict_cd"] = df["subdistrict_cd"].astype(str)
    clip = float(df["net_inflow"].abs().quantile(0.95))
    norm = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)

    fig, ax = plt.subplots(figsize=(6, 5))
    gdf.merge(df, on="subdistrict_cd", how="left").plot(
        column="net_inflow",
        cmap="RdBu",
        norm=norm,
        ax=ax,
        edgecolor="white",
        linewidth=0.25,
        legend=True,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"shrink": 0.6},
    )
    ax.axis("off")
    fig.tight_layout()
    save(fig, output_dir, "fig_01d_net_inflow.png")


def plot_24h_profile(con: duckdb.DuckDBPyConnection, data_root: Path, data_glob: str, output_dir: Path, year: int) -> None:
    calendar = pd.read_csv(data_root / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar[(calendar["date"].dt.year == year)].copy()
    calendar["day_type"] = np.where(
        calendar["is_holiday"] == 1,
        "Holiday",
        np.where(calendar["is_weekend"] == 1, "Weekend", "Weekday"),
    )
    con.register("calendar", calendar[["date", "day_type"]])
    df = con.execute(
        f"""
        SELECT c.day_type, o.st_hour, SUM(o.flow) AS total_flow
        FROM read_parquet('{data_glob}', union_by_name=true) o
        JOIN calendar c ON o.date = c.date
        WHERE {year_filter(year, "o.date")}
        GROUP BY c.day_type, o.st_hour
        """
    ).df()
    day_counts = calendar["day_type"].value_counts().to_dict()
    df["avg_flow"] = df.apply(lambda row: row["total_flow"] / day_counts[row["day_type"]], axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {"Weekday": ("-", "steelblue"), "Weekend": ("--", "coral"), "Holiday": (":", "forestgreen")}
    for day_type, (linestyle, color) in styles.items():
        sub = df[df["day_type"] == day_type].sort_values("st_hour")
        ax.plot(sub["st_hour"], sub["avg_flow"] / 1e6, linestyle=linestyle, color=color, linewidth=2.5, label=day_type)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg Daily Flow (M)")
    ax.set_xticks(range(0, 24, 4))
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "fig_02_24h_profile.png")


def plot_monthly_trend(con: duckdb.DuckDBPyConnection, data_glob: str, output_dir: Path) -> None:
    df = con.execute(
        f"""
        SELECT
            EXTRACT(YEAR FROM date) AS year,
            EXTRACT(MONTH FROM date) AS month,
            date,
            SUM(flow) AS daily_flow
        FROM read_parquet('{data_glob}', union_by_name=true)
        GROUP BY year, month, date
        """
    ).df()
    df["year"] = df["year"].astype(int).astype(str)
    df["month"] = df["month"].astype(int)
    df["hourly_avg_m"] = df["daily_flow"] / 24 / 1e6

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="month", y="hourly_avg_m", hue="year", fliersize=1.5, ax=ax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LABELS, fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Hourly Flow (M)")
    ax.legend(title="Year", loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "fig_05_monthly_boxplot.png")


def plot_mode_composition(con: duckdb.DuckDBPyConnection, data_glob: str, output_dir: Path, year: int) -> None:
    n_days = con.execute(
        f"""
        SELECT COUNT(DISTINCT date)
        FROM read_parquet('{data_glob}', union_by_name=true)
        WHERE {year_filter(year)}
        """
    ).fetchone()[0]
    df = con.execute(
        f"""
        SELECT st_hour, mode, SUM(flow) AS total_flow
        FROM read_parquet('{data_glob}', union_by_name=true)
        WHERE {year_filter(year)}
        GROUP BY st_hour, mode
        """
    ).df()
    pivot = df.pivot(index="st_hour", columns="mode", values="total_flow").reindex(columns=MODE_ORDER)
    pivot.columns = [MODE_NAMES[mode] for mode in pivot.columns]
    pivot = pivot / n_days / 1e6

    fig, ax = plt.subplots(figsize=(6, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=[MODE_COLORS[name] for name in pivot.columns], width=0.85)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg Daily Flow (M)")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([str(hour) for hour in range(0, 24, 4)], rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "fig_03_mode_composition.png")


def plot_temporal_correlation(con: duckdb.DuckDBPyConnection, data_glob: str, output_dir: Path, year: int) -> None:
    df = con.execute(
        f"""
        SELECT date, st_hour, mode, SUM(flow) AS total_flow
        FROM read_parquet('{data_glob}', union_by_name=true)
        WHERE {year_filter(year)}
        GROUP BY date, st_hour, mode
        """
    ).df()
    pivot = df.pivot_table(index=["date", "st_hour"], columns="mode", values="total_flow", fill_value=0)
    pivot = pivot.reindex(columns=MODE_ORDER)
    pivot.columns = [MODE_NAMES[mode] for mode in pivot.columns]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot.corr(), annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0, square=True, ax=ax)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save(fig, output_dir, "fig_11_temporal_corr.png")


def main() -> None:
    args = parse_args()
    setup_plot()
    data_glob = parquet_glob(args.data_root)
    con = duckdb.connect()
    plot_flow_distribution(con, data_glob, args.output_dir, args.sample_size)
    plot_net_inflow(con, args.data_root, data_glob, args.output_dir, args.year)
    plot_24h_profile(con, args.data_root, data_glob, args.output_dir, args.year)
    plot_monthly_trend(con, data_glob, args.output_dir)
    plot_mode_composition(con, data_glob, args.output_dir, args.year)
    plot_temporal_correlation(con, data_glob, args.output_dir, args.year)


if __name__ == "__main__":
    main()
