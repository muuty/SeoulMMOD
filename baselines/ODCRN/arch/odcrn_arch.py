from __future__ import annotations

import torch
from torch import nn


class ODconv(nn.Module):
    def __init__(self, K: int, input_dim: int, hidden_dim: int,
                 use_bias: bool = True, activation=None) -> None:
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.W = nn.Parameter(torch.empty(self.input_dim * (self.K ** 2), self.hidden_dim))
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.constant_(self.b, val=0.0)

    def forward(self, X: torch.Tensor, G: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feat_set = []
        if isinstance(G, tuple):
            if len(G) != 2:
                raise ValueError("Dynamic OD graph must be an (origin, destination) tuple")
            if G[0].ndim == 2 and G[1].ndim == 2:
                T_k = [[torch.eye(G[i].shape[0], device=X.device), G[i]] for i in range(2)]
                T_k_pair = []
                for i in range(2):
                    for _ in range(2, self.K):
                        T_k[i].append(2 * torch.mm(G[i], T_k[i][-1]) - T_k[i][-2])
                    T_k_pair.append(torch.stack(T_k[i], dim=0))
                G = tuple(T_k_pair)
                for o in range(self.K):
                    for d in range(self.K):
                        mode_1_prod = torch.einsum("bncl,nm->bmcl", X, G[0][o])
                        mode_2_prod = torch.einsum("bmcl,cd->bmdl", mode_1_prod, G[1][d])
                        feat_set.append(mode_2_prod)
            else:
                raise NotImplementedError
        else:
            if self.K != G.shape[-3]:
                raise ValueError(f"Expected {self.K} supports, got {G.shape[-3]}")
            for o in range(self.K):
                for d in range(self.K):
                    mode_1_prod = torch.einsum("bncl,nm->bmcl", X, G[o])
                    mode_2_prod = torch.einsum("bmcl,cd->bmdl", mode_1_prod, G[d])
                    feat_set.append(mode_2_prod)

        feat = torch.cat(feat_set, dim=-1)
        out = torch.einsum("bmdk,kh->bmdh", feat, self.W)
        if self.use_bias:
            out = out + self.b
        return self.activation(out) if self.activation is not None else out


class ODCRUcell(nn.Module):
    def __init__(self, num_nodes: int, K: int, input_dim: int, hidden_dim: int,
                 use_bias: bool = True, activation=None) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.gates = ODconv(K, input_dim + hidden_dim, hidden_dim * 2, use_bias, activation)
        self.candi = ODconv(K, input_dim + hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        weight = next(self.parameters()).data
        return weight.new_zeros(batch_size, self.num_nodes, self.num_nodes, self.hidden_dim)

    def forward(self, G: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                Xt: torch.Tensor, Ht_1: torch.Tensor) -> torch.Tensor:
        XH = torch.cat([Xt, Ht_1], dim=-1)
        XH_conv = self.gates(X=XH, G=G)
        u, r = torch.split(XH_conv, self.hidden_dim, dim=-1)
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)
        candi = torch.cat([Xt, reset * Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi, G=G))
        return (1.0 - update) * Ht_1 + update * candi_conv


class ODCRUencoder(nn.Module):
    def __init__(self, num_nodes: int, K: int, input_dim: int, hidden_dim: int,
                 num_layers: int, use_bias: bool = True, activation=None,
                 return_all_layers: bool = True) -> None:
        super().__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        if len(self.hidden_dim) != self.num_layers:
            raise ValueError("hidden_dim and num_layers are inconsistent")
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(
                ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias, activation)
            )

    def forward(self, G_list: list, X_seq: torch.Tensor, H0_l=None):
        if self.num_layers != len(G_list):
            raise ValueError("num_layers must match G_list length")
        batch_size, seq_len, _, _, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)

        out_seq_lst = []
        Ht_lst = []
        in_seq_l = X_seq
        for layer_idx in range(self.num_layers):
            Ht = H0_l[layer_idx]
            out_seq_l = []
            for t in range(seq_len):
                graph = G_list[layer_idx]
                if not isinstance(graph, (torch.Tensor, tuple)):
                    graph = graph(x_t=X_seq[:, t])
                Ht = self.cell_list[layer_idx](G=graph, Xt=in_seq_l[:, t], Ht_1=Ht)
                out_seq_l.append(Ht)
            out_seq_l = torch.stack(out_seq_l, dim=1)
            in_seq_l = out_seq_l
            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            Ht_lst = Ht_lst[-1:]
        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size: int) -> list[torch.Tensor]:
        return [cell.init_hidden(batch_size) for cell in self.cell_list]

    @staticmethod
    def _extend_for_multilayers(param, num_layers: int):
        return param if isinstance(param, list) else [param] * num_layers


class ODCRUdecoder(nn.Module):
    def __init__(self, num_nodes: int, K: int, output_dim: int, hidden_dim: int,
                 out_horizon: int, num_layers: int, use_bias: bool = True,
                 activation=None) -> None:
        super().__init__()
        self.out_horizon = out_horizon
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        if len(self.hidden_dim) != self.num_layers:
            raise ValueError("hidden_dim and num_layers are inconsistent")
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(
                ODCRUcell(num_nodes, K, cur_input_dim, self.hidden_dim[i], use_bias, activation)
            )

    def forward(self, G_list: list, Xt: torch.Tensor, H0_l: list[torch.Tensor]):
        if self.num_layers != len(G_list):
            raise ValueError("num_layers must match G_list length")
        Ht_lst = []
        Xin_l = Xt
        for layer_idx in range(self.num_layers):
            graph = G_list[layer_idx]
            if not isinstance(graph, torch.Tensor):
                graph = graph(x_t=Xt)
            Ht_l = self.cell_list[layer_idx](G=graph, Xt=Xin_l, Ht_1=H0_l[layer_idx])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l
        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers: int):
        return param if isinstance(param, list) else [param] * num_layers


class DynGraphConstructor(nn.Module):
    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.W_o = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.W_d = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.W_d)

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        O = torch.softmax(torch.relu(torch.einsum("bpdh,dd,bqdh->pq", x_t, self.W_o.to(x_t.device), x_t)), dim=1)
        D = torch.softmax(torch.relu(torch.einsum("boeh,oo,bofh->ef", x_t, self.W_d.to(x_t.device), x_t)), dim=1)
        return O, D


class ODCRN(nn.Module):
    def __init__(self, num_nodes: int, K: int, input_dim: int, hidden_dim: int,
                 out_horizon: int, num_layers: int, DGCbool: bool = True,
                 use_bias: bool = True, activation=None) -> None:
        super().__init__()
        self.DGCbool = DGCbool
        if self.DGCbool:
            self.DGC = DynGraphConstructor(num_nodes)
        self.encoder = ODCRUencoder(num_nodes, K, input_dim, hidden_dim, num_layers,
                                    use_bias, activation, return_all_layers=True)
        self.decoder = ODCRUdecoder(num_nodes, K, input_dim, hidden_dim, out_horizon,
                                    num_layers, use_bias, activation)
        self.linear = nn.Linear(hidden_dim, input_dim, bias=use_bias)

    def forward(self, G: torch.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        if self.DGCbool:
            _, Ht_lst = self.encoder(G_list=[self.DGC, G], X_seq=X_seq, H0_l=None)
        else:
            _, Ht_lst = self.encoder(G_list=[G, G], X_seq=X_seq, H0_l=None)

        deco_input = torch.zeros(X_seq.shape[:1] + X_seq.shape[2:], device=X_seq.device)
        outputs = []
        for _ in range(self.decoder.out_horizon):
            if self.DGCbool:
                Ht_l, Ht_lst = self.decoder(G_list=[self.DGC, G], Xt=deco_input, H0_l=Ht_lst)
            else:
                Ht_l, Ht_lst = self.decoder(G_list=[G, G], Xt=deco_input, H0_l=Ht_lst)
            output = self.linear(Ht_l)
            deco_input = output
            outputs.append(output)
        return torch.stack(outputs, dim=1)
