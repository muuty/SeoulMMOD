"""STIDPlus: STID base + optional context embeddings (holiday, POI, dist/time channels).

Each context source is toggled via a flag and contributes one extra block to the
encoder hidden representation (concat along channel dim, identical to how STID
already adds time-of-day, day-of-week, and node embeddings).

Holiday: per-timestep categorical state (workday=0 / weekend=1 / holiday=2).
  Activated by ``if_holiday=True`` and ``holiday_index=<channel>``. Stored in the
  dataset as ``state / holiday_size`` (recovered via ``(val * size).long()``).

POI: per-node static features (e.g. 13 district-scale POI categories), looked up from
  ``poi_features_path`` (CSV) at __init__ and combined into a per-pair feature
  via ``concat(origin_poi, dest_poi)``. Activated by ``if_poi=True``.

Distance/time (or any extra dynamic channel): each gets its own Conv2d
  projection so it doesn't share capacity with the mode-flow projection.
  Configure via ``extra_input_specs`` - a list of dicts:
      [{"name": "dist", "ch": <idx_in_FORWARD_FEATURES>, "emb_dim": 16},
       {"name": "time", "ch": ..., "emb_dim": 16}]
  Each emb is concatenated to the encoder hidden as its own block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .mlp import MultiLayerPerceptron


def _build_poi_pair_features(poi_features_path: str, dataset_name: str,
                             num_nodes: int, poi_categories: list[str],
                             repo_root: Path) -> np.ndarray:
    """Return (num_nodes, 2 * len(poi_categories)) per-pair feature matrix.

    Each pair gets concat(origin_poi, dest_poi). Pair order must match the
    dataset's node axis (i.e. order in od_pairs.json).
    """
    od_path = repo_root / 'datasets' / dataset_name / 'od_pairs.json'
    with open(od_path) as f:
        od = json.load(f)
    pairs = od['od_pairs']
    if len(pairs) != num_nodes:
        raise ValueError(f'pair count mismatch: dataset N={num_nodes} vs od_pairs len={len(pairs)}')

    df = pd.read_csv(poi_features_path)
    code_col = 'district_cd'
    if code_col not in df.columns:
        raise ValueError(f'district features file missing district_cd column: {poi_features_path}')
    df[code_col] = df[code_col].astype(int)
    poi_by_district = {row[code_col]: row[poi_categories].to_numpy(dtype=np.float32)
                       for _, row in df.iterrows()}

    K = len(poi_categories)
    feats = np.zeros((num_nodes, 2 * K), dtype=np.float32)
    for idx, (o, d) in enumerate(pairs):
        o_int = int(o)
        d_int = int(d)
        if o_int not in poi_by_district or d_int not in poi_by_district:
            raise ValueError(f'pair {idx} ({o},{d}): district not in POI table')
        feats[idx, :K] = poi_by_district[o_int]
        feats[idx, K:] = poi_by_district[d_int]
    return feats


class STIDPlus(nn.Module):
    """STID with optional holiday and POI context embeddings."""

    def __init__(self, **model_args):
        super().__init__()
        # ----- core STID args -----
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_dim = model_args.get("output_dim", 1)
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.tod_index = model_args.get("tod_index", 1)
        self.dow_index = model_args.get("dow_index", 2)

        # ----- context flags -----
        self.if_holiday = model_args.get("if_holiday", False)
        self.if_poi = model_args.get("if_poi", False)
        self.extra_input_specs = list(model_args.get("extra_input_specs", []))

        # core embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # holiday embedding(s). Two independent lookups are supported:
        #   - input window start  (history_data[:, 0, :, holiday_idx])
        #   - prediction window start (future_data[:, 0, :, holiday_idx])
        # ``if_holiday`` (legacy) uses the input window's LAST hour for backward
        # compat; set ``holiday_use_input_start=True`` to use the first hour
        # instead (recommended when pairing with ``if_holiday_future``).
        self.holiday_use_input_start = model_args.get("holiday_use_input_start", False)
        self.if_holiday_future = model_args.get("if_holiday_future", False)
        self.holiday_size = model_args.get("holiday_size", 3)
        self.temp_dim_holiday = 0
        if self.if_holiday:
            self.holiday_index = model_args["holiday_index"]
            self.temp_dim_holiday_in = model_args.get("temp_dim_holiday", 32)
            self.holiday_emb_in = nn.Parameter(
                torch.empty(self.holiday_size, self.temp_dim_holiday_in))
            nn.init.xavier_uniform_(self.holiday_emb_in)
            self.temp_dim_holiday += self.temp_dim_holiday_in
        if self.if_holiday_future:
            self.holiday_index = model_args["holiday_index"]
            self.temp_dim_holiday_out = model_args.get("temp_dim_holiday_future", 32)
            self.holiday_emb_out = nn.Parameter(
                torch.empty(self.holiday_size, self.temp_dim_holiday_out))
            nn.init.xavier_uniform_(self.holiday_emb_out)
            self.temp_dim_holiday += self.temp_dim_holiday_out

        # POI embedding (static per-pair feature, projected to fixed dim)
        if self.if_poi:
            poi_features_path = model_args["poi_features_path"]
            poi_categories = model_args["poi_categories"]
            self.poi_dim = model_args.get("poi_dim", 32)
            poi_hidden = model_args.get("poi_hidden", 64)
            dataset_name = model_args["dataset_name"]
            repo_root = Path(model_args.get("repo_root",
                              Path(__file__).resolve().parents[3]))
            poi_pair = _build_poi_pair_features(
                poi_features_path, dataset_name, self.num_nodes,
                poi_categories, repo_root,
            )
            # Log1p to compress heavy tail, then per-feature z-score for stable training.
            poi_pair = np.log1p(poi_pair)
            mu = poi_pair.mean(axis=0, keepdims=True)
            sd = poi_pair.std(axis=0, keepdims=True) + 1e-6
            poi_pair = (poi_pair - mu) / sd
            self.register_buffer("poi_raw",
                                 torch.from_numpy(poi_pair.astype(np.float32)))
            in_dim = poi_pair.shape[1]
            self.poi_proj = nn.Sequential(
                nn.Linear(in_dim, poi_hidden),
                nn.ReLU(),
                nn.Linear(poi_hidden, self.poi_dim),
            )
        else:
            self.poi_dim = 0

        # input projection (only mode channels go through this)
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # one Conv2d per extra dynamic input (e.g., dist, time)
        self.extra_input_layers = nn.ModuleDict()
        for spec in self.extra_input_specs:
            self.extra_input_layers[spec["name"]] = nn.Conv2d(
                in_channels=self.input_len,
                out_channels=spec["emb_dim"],
                kernel_size=(1, 1),
                bias=True,
            )

        sum_extra = sum(spec["emb_dim"] for spec in self.extra_input_specs)
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
            + self.temp_dim_holiday  # already 0 if neither flag set
            + self.poi_dim * int(self.if_poi)
            + sum_extra
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(self.num_layer)]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len * self.output_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # history_data: [B, L, N, F_in]; only first ``input_dim`` channels are mode flows.
        input_data = history_data[..., :self.input_dim]

        if self.if_time_in_day:
            t = history_data[..., self.tod_index]
            time_in_day_emb = self.time_in_day_emb[
                (t[:, -1, :] * self.time_of_day_size).long()
            ]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d = history_data[..., self.dow_index]
            day_in_week_emb = self.day_in_week_emb[
                (d[:, -1, :] * self.day_of_week_size).long()
            ]
        else:
            day_in_week_emb = None
        if self.if_holiday:
            h = history_data[..., self.holiday_index]
            # Default: last hour (legacy STID convention). Set
            # holiday_use_input_start=True to use first hour ("input window start").
            t_idx = 0 if self.holiday_use_input_start else -1
            holiday_emb_in = self.holiday_emb_in[
                (h[:, t_idx, :] * self.holiday_size).long().clamp(0, self.holiday_size - 1)
            ]
        else:
            holiday_emb_in = None
        if self.if_holiday_future:
            # First hour of prediction window (i.e., "prediction start" state).
            hf = future_data[..., self.holiday_index]
            holiday_emb_out = self.holiday_emb_out[
                (hf[:, 0, :] * self.holiday_size).long().clamp(0, self.holiday_size - 1)
            ]
        else:
            holiday_emb_out = None

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # Extra dynamic inputs (e.g., dist, time): each gets its own Conv2d.
        # Per channel: history_data[..., ch] -> (B, L, N) -> (B, L, N, 1) Conv2d input -> (B, emb_dim, N, 1)
        extra_emb_blocks = []
        for spec in self.extra_input_specs:
            ch = spec["ch"]
            extra_data = history_data[..., ch]                      # (B, L, N)
            extra_data = extra_data.unsqueeze(-1)                    # (B, L, N, 1)
            layer = self.extra_input_layers[spec["name"]]
            extra_emb_blocks.append(layer(extra_data))               # (B, emb_dim, N, 1)

        node_emb = []
        if self.if_spatial:
            node_emb.append(
                self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
                .transpose(1, 2).unsqueeze(-1)
            )
        if self.if_poi:
            poi = self.poi_proj(self.poi_raw)  # (N, poi_dim)
            poi = poi.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            node_emb.append(poi)

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        if holiday_emb_in is not None:
            tem_emb.append(holiday_emb_in.transpose(1, 2).unsqueeze(-1))
        if holiday_emb_out is not None:
            tem_emb.append(holiday_emb_out.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + extra_emb_blocks + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)

        prediction = self.regression_layer(hidden)
        if self.output_dim > 1:
            B, _, N, _ = prediction.shape
            prediction = prediction.view(B, self.output_len, self.output_dim, N).permute(0, 1, 3, 2)
        return prediction
