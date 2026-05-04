from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


MODE_NAMES = {
    4: "Metro Bus",
    5: "Local Bus",
    6: "Subway",
    7: "Walk",
    8: "Car",
    9: "Other",
}
MODE_COLORS = {
    4: "#FF9F40",
    5: "#FFCF40",
    6: "#E73C3C",
    7: "#4CAF50",
    8: "#2196F3",
    9: "#9C27B0",
}
MODE_ORDER = [4, 5, 6, 7, 8, 9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--year", type=int, default=2024)
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


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=13)


def load_travel_time_outflow(con: duckdb.DuckDBPyConnection, data_glob: str, year: int) -> pd.DataFrame:
    df = con.execute(
        f"""
        WITH valid AS (
            SELECT
                SUBSTR(origin, 1, 5) AS district_cd,
                mode,
                date,
                st_hour,
                flow,
                avg_time_min,
                avg_dist_m
            FROM read_parquet('{data_glob}', union_by_name=true)
            WHERE {year_filter(year)}
              AND flow > 0
              AND mode IN (4, 5, 6, 7, 8, 9)
              AND avg_time_min > 0 AND avg_time_min <= 180
              AND avg_dist_m > 0
              AND (avg_dist_m / 1000.0) / (avg_time_min / 60.0) BETWEEN 0.5 AND 150
        ),
        hours AS (
            SELECT COUNT(*) AS n_hours
            FROM (SELECT DISTINCT date, st_hour FROM valid)
        )
        SELECT
            district_cd,
            mode,
            SUM(flow) / (SELECT n_hours FROM hours) AS avg_hourly_outflow,
            SUM(flow * avg_time_min) / SUM(flow) AS avg_time_min,
            SUM(flow * avg_dist_m) / SUM(flow) / 1000.0 AS avg_dist_km
        FROM valid
        GROUP BY district_cd, mode
        """
    ).df()
    df["mode"] = df["mode"].astype(int)
    df["mode_name"] = df["mode"].map(MODE_NAMES)
    return df


def load_rain_change(con: duckdb.DuckDBPyConnection, data_root: Path, data_glob: str, year: int) -> pd.DataFrame:
    rain = pd.concat(
        [pd.read_csv(path) for path in sorted((data_root / "rainfall").glob(f"seoul_rain_hourly_{year}*.csv"))],
        ignore_index=True,
    )
    rain["timestamp"] = pd.to_datetime(rain["timestamp"])
    rain["date"] = rain["timestamp"].dt.normalize()
    rain["st_hour"] = rain["timestamp"].dt.hour
    hourly = rain.groupby(["date", "st_hour"], as_index=False)["rain_mm"].mean()
    hourly["rain_cat"] = np.select(
        [
            hourly["rain_mm"] == 0,
            hourly["rain_mm"] < 0.5,
            hourly["rain_mm"] < 2,
            hourly["rain_mm"] <= 10,
        ],
        ["Dry", "Trace", "Light", "Moderate"],
        default="Heavy",
    )
    hourly = hourly[hourly["rain_cat"] != "Heavy"].copy()
    con.register("rain_hourly", hourly[["date", "st_hour", "rain_cat"]])
    df = con.execute(
        f"""
        WITH hour_mode AS (
            SELECT date, st_hour, mode, SUM(flow) AS flow
            FROM read_parquet('{data_glob}', union_by_name=true)
            WHERE {year_filter(year)}
            GROUP BY date, st_hour, mode
        )
        SELECT r.rain_cat, h.mode, AVG(h.flow) AS avg_hour_flow
        FROM hour_mode h
        JOIN rain_hourly r ON h.date = r.date AND h.st_hour = r.st_hour
        GROUP BY r.rain_cat, h.mode
        """
    ).df()
    baseline = df[df["rain_cat"] == "Dry"].set_index("mode")["avg_hour_flow"]
    df["change_pct"] = df.apply(lambda row: (row["avg_hour_flow"] / baseline.loc[row["mode"]] - 1) * 100, axis=1)
    df["mode"] = df["mode"].astype(int)
    df["mode_name"] = df["mode"].map(MODE_NAMES)
    return df


def load_area_inflow(con: duckdb.DuckDBPyConnection, data_root: Path, data_glob: str, year: int) -> pd.DataFrame:
    areas = pd.DataFrame(
        {
            "dest": [
                "11560540",
                "11680580",
                "11680590",
                "11650531",
                "11140550",
                "11170625",
                "11710710",
                "11650581",
            ],
            "area": ["Office"] * 4 + ["Leisure"] * 4,
        }
    )
    calendar = pd.read_csv(data_root / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar[(calendar["date"].dt.year == year) & ((calendar["is_workday"] == 1) | (calendar["is_holiday"] == 1))]
    calendar["day_type"] = np.where(calendar["is_holiday"] == 1, "Holiday", "Weekday")
    con.register("areas", areas)
    con.register("day_type", calendar[["date", "day_type"]])
    return con.execute(
        f"""
        WITH hourly AS (
            SELECT t.date, t.st_hour, a.area, SUM(t.flow) AS inflow
            FROM read_parquet('{data_glob}', union_by_name=true) t
            JOIN areas a ON t.dest = a.dest
            JOIN day_type d ON t.date = d.date
            WHERE {year_filter(year, "t.date")}
            GROUP BY t.date, t.st_hour, a.area
        )
        SELECT h.area, d.day_type, h.st_hour, AVG(h.inflow) AS mean_inflow
        FROM hourly h
        JOIN day_type d ON h.date = d.date
        GROUP BY h.area, d.day_type, h.st_hour
        """
    ).df()


def load_poi_inflow(con: duckdb.DuckDBPyConnection, data_root: Path, data_glob: str, year: int) -> pd.DataFrame:
    picks = pd.DataFrame(
        [
            {"code": "11560540", "category": "Office"},
            {"code": "11620735", "category": "Education"},
            {"code": "11680660", "category": "Residential"},
        ]
    )
    calendar = pd.read_csv(data_root / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    workdays = calendar[(calendar["date"].dt.year == year) & (calendar["is_workday"] == 1)][["date"]]
    con.register("selected", picks)
    con.register("workdays", workdays)
    return con.execute(
        f"""
        WITH hours AS (
            SELECT s.code, s.category, w.date, h.st_hour
            FROM selected s
            CROSS JOIN workdays w
            CROSS JOIN (SELECT range AS st_hour FROM range(24)) h
        ),
        inflow AS (
            SELECT dest AS code, date, st_hour, SUM(flow) AS inflow
            FROM read_parquet('{data_glob}', union_by_name=true)
            WHERE {year_filter(year)} AND dest IN (SELECT code FROM selected)
            GROUP BY dest, date, st_hour
        ),
        outflow AS (
            SELECT origin AS code, date, st_hour, SUM(flow) AS outflow
            FROM read_parquet('{data_glob}', union_by_name=true)
            WHERE {year_filter(year)} AND origin IN (SELECT code FROM selected)
            GROUP BY origin, date, st_hour
        )
        SELECT
            h.category,
            h.st_hour,
            AVG((COALESCE(i.inflow, 0) - COALESCE(o.outflow, 0))
                / NULLIF(COALESCE(i.inflow, 0) + COALESCE(o.outflow, 0), 0)) AS net_ratio
        FROM hours h
        LEFT JOIN inflow i USING (code, date, st_hour)
        LEFT JOIN outflow o USING (code, date, st_hour)
        GROUP BY h.category, h.st_hour
        """
    ).df()


def plot_travel_time_outflow(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sizes = np.sqrt(df["avg_hourly_outflow"].clip(lower=1)) * 1.6
    for mode in MODE_ORDER:
        sub = df[df["mode"] == mode]
        ax.scatter(
            sub["avg_dist_km"],
            sub["avg_time_min"],
            s=sizes.loc[sub.index],
            color=MODE_COLORS[mode],
            alpha=0.45,
            edgecolor="white",
            linewidth=0.25,
            label=MODE_NAMES[mode],
        )
    ax.set_xlabel("Avg Travel Distance (km)")
    ax.set_ylabel("Avg Travel Time (min)")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=11)
    style_axis(ax)
    save(fig, output_dir, "fig_15a_travel_time_outflow")


def plot_rain(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ["Dry", "Trace", "Light", "Moderate"]
    modes = ["Local Bus", "Subway", "Walk"]
    palette = {"Local Bus": MODE_COLORS[5], "Subway": MODE_COLORS[6], "Walk": MODE_COLORS[7]}
    sns.barplot(
        data=df[df["mode_name"].isin(modes)],
        x="rain_cat",
        y="change_pct",
        hue="mode_name",
        order=order,
        hue_order=modes,
        palette=palette,
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Rainfall category")
    ax.set_ylabel("Flow Change (%)")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    save(fig, output_dir, "fig_15b_rainfall_by_mode")


def plot_area_inflow(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"Office": "#1f77b4", "Leisure": "#d62728"}
    linestyles = {"Weekday": "-", "Holiday": "--"}
    for area in ["Office", "Leisure"]:
        for day_type in ["Weekday", "Holiday"]:
            sub = df[(df["area"] == area) & (df["day_type"] == day_type)].sort_values("st_hour")
            ax.plot(sub["st_hour"], sub["mean_inflow"] / 1000, color=palette[area], linestyle=linestyles[day_type], linewidth=2.5)
    ax.legend(
        handles=[
            Line2D([0], [0], color=palette["Office"], linewidth=2.5, label="Office"),
            Line2D([0], [0], color=palette["Leisure"], linewidth=2.5, label="Leisure"),
            Line2D([0], [0], color="#333333", linewidth=2.2, linestyle="-", label="Weekday"),
            Line2D([0], [0], color="#333333", linewidth=2.2, linestyle="--", label="Holiday"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=12,
    )
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg Inflow (K)")
    style_axis(ax)
    save(fig, output_dir, "fig_15c_calendar_shift")


def plot_poi_inflow(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"Office": "#1f77b4", "Education": "#2ca02c", "Residential": "#7f7f7f"}
    for category, color in palette.items():
        sub = df[df["category"] == category].sort_values("st_hour")
        ax.plot(sub["st_hour"], sub["net_ratio"], color=color, linewidth=2.5, marker="o", markersize=4.8, label=category)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Net Inflow Ratio")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    save(fig, output_dir, "fig_15d_poi_linked_inflow")


def main() -> None:
    args = parse_args()
    setup_plot()
    con = duckdb.connect()
    data_glob = parquet_glob(args.data_root)
    plot_travel_time_outflow(load_travel_time_outflow(con, data_glob, args.year), args.output_dir)
    plot_rain(load_rain_change(con, args.data_root, data_glob, args.year), args.output_dir)
    plot_area_inflow(load_area_inflow(con, args.data_root, data_glob, args.year), args.output_dir)
    plot_poi_inflow(load_poi_inflow(con, args.data_root, data_glob, args.year), args.output_dir)


if __name__ == "__main__":
    main()
