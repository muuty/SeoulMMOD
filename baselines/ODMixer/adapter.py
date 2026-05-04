from __future__ import annotations

import torch
from torch import nn

from .arch.odmixer_arch import ODMixerBackbone


class ODMixerAdapter(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 16,
        layer_nums: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.backbone = ODMixerBackbone(
            num_nodes=num_nodes,
            input_seq=input_len * input_dim,
            hidden_dim=hidden_dim,
            layer_nums=layer_nums,
            dropout=dropout,
            out_steps=output_len * output_dim,
        )

    def _fold_modes(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, _, _, input_dim = data.shape
        return data.permute(0, 1, 4, 2, 3).reshape(
            batch_size, input_len * input_dim, self.num_nodes, self.num_nodes
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool,
                prev_history_data: torch.Tensor | None = None,
                prev_future_data: torch.Tensor | None = None,
                **kwargs) -> dict:
        batch_size = history_data.shape[0]
        x = self._fold_modes(history_data)
        prev_x = self._fold_modes(prev_history_data)
        od_out, prev_out = self.backbone(x, prev_x)
        prediction = od_out.reshape(
            batch_size, self.output_len, self.output_dim, self.num_nodes, self.num_nodes
        ).permute(0, 1, 3, 4, 2).contiguous()
        prev_prediction = prev_out.reshape(
            batch_size, self.output_len, self.output_dim, self.num_nodes, self.num_nodes
        ).permute(0, 1, 3, 4, 2).contiguous()
        return {"prediction": prediction, "prev_prediction": prev_prediction}
