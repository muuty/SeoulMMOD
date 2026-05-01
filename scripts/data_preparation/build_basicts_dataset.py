"""Build SeoulMMOD BasicTS datasets from processed parquet files.

The input parquet files are expected to contain hourly Seoul-internal OD
flows after raw-data cleaning and administrative-code harmonization.

Expected parquet columns:
    date, st_hour, origin, dest, mode, flow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np

MODE_CODES = [4, 5, 6, 7, 8, 9]
MODE_NAMES = ["metro_bus", "local_bus", "subway", "walk", "car", "other"]

STEPS_PER_DAY = 24


def parquet_union(input_dir: Path, years: list[int]) -> str:
    paths = [input_dir / f"od_flow_{year}.parquet" for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing parquet files: " + ", ".join(missing))
    return " UNION ALL ".join(
        f"SELECT * FROM '{str(path).replace(chr(39), chr(39) + chr(39))}'"
        for path in paths
    )


def default_dataset_name(level: str, years: list[int]) -> str:
    parts = ["SeoulMOD"]
    if level == "gu":
        parts.append("GU")
    if len(years) == 1:
        parts.append(str(years[0]))
    else:
        parts.append(f"{min(years)}-{max(years)}")
    return "_".join(parts)


def od_expr(level: str) -> tuple[str, str]:
    if level == "gu":
        return "SUBSTRING(origin, 1, 5)", "SUBSTRING(dest, 1, 5)"
    return "origin", "dest"


def write_desc(
    out_dir: Path,
    name: str,
    level: str,
    feature_names: list[str],
    start_date: object,
    end_date: object,
    shape: tuple[int, int, int],
) -> None:
    t_steps, n_nodes, n_features = shape
    desc = {
        "name": name,
        "domain": f"multimodal OD flow ({level}-level)",
        "shape": [t_steps, n_nodes, n_features],
        "num_time_steps": t_steps,
        "num_nodes": n_nodes,
        "num_features": n_features,
        "feature_description": feature_names + ["time_of_day", "day_of_week"],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "frequency (minutes)": 60,
        "regular_settings": {
            "INPUT_LEN": 24,
            "OUTPUT_LEN": 24,
            "TRAIN_VAL_TEST_RATIO": [0.7, 0.1, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE"],
            "NULL_VAL": None,
        },
    }
    with open(out_dir / "desc.json", "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)


def fill_calendar_channels(
    fp: np.memmap, day_idx: int, day: object, n_flow_channels: int
) -> None:
    day_of_week = day.weekday() / 7.0
    for hour in range(STEPS_PER_DAY):
        t_idx = day_idx * STEPS_PER_DAY + hour
        fp[t_idx, :, n_flow_channels] = hour / 24.0
        fp[t_idx, :, n_flow_channels + 1] = day_of_week


def build_mode_dataset(
    con: duckdb.DuckDBPyConnection,
    dataset_sql: str,
    out_dir: Path,
    dataset_name: str,
    level: str,
) -> None:
    origin_expr, dest_expr = od_expr(level)
    od_pairs = con.execute(
        f"""
        SELECT DISTINCT {origin_expr} AS o, {dest_expr} AS d
        FROM ({dataset_sql})
        ORDER BY o, d
        """
    ).fetchall()
    origins = [row[0] for row in od_pairs]
    dests = [row[1] for row in od_pairs]
    pair_to_idx = {(o, d): idx for idx, (o, d) in enumerate(od_pairs)}
    mode_to_idx = {mode: idx for idx, mode in enumerate(MODE_CODES)}

    start_date, end_date, n_days = con.execute(
        f"SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM ({dataset_sql})"
    ).fetchone()
    t_steps = n_days * STEPS_PER_DAY
    n_nodes = len(od_pairs)
    n_features = len(MODE_CODES) + 2

    out_dir.mkdir(parents=True, exist_ok=True)
    fp = np.memmap(
        str(out_dir / "data.dat"),
        dtype="float32",
        mode="w+",
        shape=(t_steps, n_nodes, n_features),
    )

    dates = [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT date FROM ({dataset_sql}) ORDER BY date"
        ).fetchall()
    ]
    for day_idx, day in enumerate(dates):
        rows = con.execute(
            f"""
            SELECT st_hour, {origin_expr} AS o, {dest_expr} AS d, mode, SUM(flow) AS flow
            FROM ({dataset_sql})
            WHERE date = DATE '{day}'
            GROUP BY st_hour, o, d, mode
            """
        ).fetchall()
        for st_hour, origin, dest, mode, flow in rows:
            pair_idx = pair_to_idx.get((origin, dest))
            mode_idx = mode_to_idx.get(mode)
            if pair_idx is None or mode_idx is None:
                continue
            t_idx = day_idx * STEPS_PER_DAY + int(st_hour)
            fp[t_idx, pair_idx, mode_idx] += float(flow)
        fill_calendar_channels(fp, day_idx, day, len(MODE_CODES))

    fp.flush()
    del fp

    write_desc(
        out_dir,
        dataset_name,
        level,
        MODE_NAMES,
        start_date,
        end_date,
        (t_steps, n_nodes, n_features),
    )
    with open(out_dir / "od_pairs.json", "w", encoding="utf-8") as f:
        json.dump(
            {"od_pairs": list(zip(origins, dests)), "modes": MODE_NAMES, "mode_codes": MODE_CODES},
            f,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("raw/parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--level", choices=["gu", "dong"], required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2024])
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    dataset_name = args.name or default_dataset_name(args.level, args.years)
    out_dir = args.output_dir / dataset_name
    dataset_sql = parquet_union(args.input_dir, args.years)

    con = duckdb.connect()
    build_mode_dataset(con, dataset_sql, out_dir, dataset_name, args.level)
    con.close()
    print(f"Saved {dataset_name} to {out_dir}")


if __name__ == "__main__":
    main()
