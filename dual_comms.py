import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # adj: [B, N, N]
    deg = adj.sum(dim=-1)  # [B, N]
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_left = deg_inv_sqrt.unsqueeze(-1)   # [B, N, 1]
    D_right = deg_inv_sqrt.unsqueeze(-2)  # [B, 1, N]
    return D_left * adj * D_right


class DualGraphComm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # same branch
        self.same_fc1 = nn.Linear(d_model, d_model)
        self.same_fc2 = nn.Linear(d_model, d_model)
        self.same_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # conflict branch
        self.conf_q = nn.Linear(d_model, d_model, bias=False)
        self.conf_k = nn.Linear(d_model, d_model, bias=False)
        self.conf_v = nn.Linear(d_model, d_model, bias=False)

        self.conf_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
        )

        self.norm_same = nn.LayerNorm(d_model)
        self.norm_conf = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def masked_attention(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, N, D]
        adj: [B, N, N], 0/1 mask
        """
        q = self.conf_q(x)
        k = self.conf_k(x)
        v = self.conf_v(x)

        mask = adj > 0
        has_nbr = mask.any(dim=-1, keepdim=True)  # [B, N, 1]

        # float32 for numerical stability
        scores = torch.bmm(q.float(), k.float().transpose(1, 2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(has_nbr, attn, torch.zeros_like(attn))

        out = torch.bmm(attn, v.float())
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor, conflict_index: torch.Tensor, same_index: torch.Tensor):
        """
        x:              [B, N, D]
        conflict_index: [B, N, N]
        same_index:     [B, N, N]
        return:         [B, N, D]
        """
        B, N, D = x.shape
        device = x.device

        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        # --------------------------------------------------
        # 1) SAME branch: attraction / consistency / cluster
        # --------------------------------------------------
        A_same = symmetric_normalize(same_index + I)

        x_same = F.relu(self.same_fc1(x))
        same_1 = torch.bmm(A_same, x_same)          # 1-hop
        same_2 = torch.bmm(A_same, same_1)          # 2-hop

        gate_same = self.same_gate(torch.cat([x, same_1, same_2], dim=-1))
        msg_same = gate_same * same_1 + (1.0 - gate_same) * self.same_fc2(same_2)

        h_same = self.norm_same(x + msg_same)

        # --------------------------------------------------
        # 2) CONFLICT branch: anti-coordination / differentiation
        # --------------------------------------------------
        # Use masked attention on conflict neighbors
        conf_nbr = self.masked_attention(x, conflict_index)

        delta_conf = x - conf_nbr
        prod_conf = x * conf_nbr

        msg_conf = self.conf_mlp(torch.cat([x, conf_nbr, delta_conf, prod_conf], dim=-1))
        h_conf = self.norm_conf(x + msg_conf)

        # --------------------------------------------------
        # 3) FUSION
        # --------------------------------------------------
        gates = self.fuse(torch.cat([x, h_same, h_conf], dim=-1))
        alpha, beta = torch.chunk(gates, 2, dim=-1)
        alpha = torch.sigmoid(alpha)
        beta = torch.sigmoid(beta)

        out = self.norm_out(x + alpha * msg_same + beta * msg_conf)
        return out
    

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def symmetric_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """
#     adj: [B, N, N]
#     """
#     adj = adj.float()
#     deg = adj.sum(dim=-1)  # [B, N]
#     deg_inv_sqrt = torch.pow(deg + eps, -0.5)
#     D_left = deg_inv_sqrt.unsqueeze(-1)   # [B, N, 1]
#     D_right = deg_inv_sqrt.unsqueeze(-2)  # [B, 1, N]
#     return D_left * adj * D_right


# class DualGraphComm(nn.Module):
#     """
#     Version A:
#       - same branch: JK-style multi-hop fusion
#       - conflict branch: GATv2-style dynamic attention
#       - fusion: alpha/beta gate

#     Input:
#         x              : [B, N, D]
#         conflict_index : [B, N, N]
#         same_index     : [B, N, N]
#     Output:
#         out            : [B, N, D]
#     """
#     def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         self.dropout = dropout

#         # -------------------------------
#         # same branch: JK-style multi-hop
#         # -------------------------------
#         self.same_fc0 = nn.Linear(d_model, d_model)
#         self.same_fc1 = nn.Linear(d_model, d_model)
#         self.same_fc2 = nn.Linear(d_model, d_model)

#         # node-wise 3-way hop attention
#         self.same_jk_gate = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, 3),
#         )
#         self.same_out = nn.Linear(d_model, d_model)

#         # -----------------------------------------
#         # conflict branch: GATv2-style dense attention
#         # -----------------------------------------
#         # pairwise transformed attention features
#         self.conf_pair_proj = nn.Linear(2 * d_model, num_heads * self.head_dim, bias=False)
#         self.conf_att = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
#         nn.init.xavier_uniform_(self.conf_att)

#         self.conf_v = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
#         self.conf_out = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

#         self.conf_mlp = nn.Sequential(
#             nn.Linear(d_model * 4, d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model),
#         )

#         # -------------------------------
#         # fusion
#         # -------------------------------
#         self.fuse = nn.Sequential(
#             nn.Linear(d_model * 3, d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model * 2),
#         )

#         self.norm_same = nn.LayerNorm(d_model)
#         self.norm_conf = nn.LayerNorm(d_model)
#         self.norm_out = nn.LayerNorm(d_model)

#     def conflict_gatv2_attention(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
#         """
#         x:   [B, N, D]
#         adj: [B, N, N], can be binary or weighted
#         returns:
#             conf_nbr: [B, N, D]
#         """
#         B, N, D = x.shape
#         device = x.device
#         dtype = x.dtype

#         # mask / weights
#         mask = adj > 0                       # [B, N, N]
#         has_nbr = mask.any(dim=-1, keepdim=True)  # [B, N, 1]

#         # float32 for numerical stability
#         x_fp32 = x.float()
#         adj_fp32 = adj.float()

#         # pairwise dynamic attention feature
#         xi = x_fp32.unsqueeze(2).expand(B, N, N, D)
#         xj = x_fp32.unsqueeze(1).expand(B, N, N, D)
#         pair = torch.cat([xi, xj], dim=-1)  # [B, N, N, 2D]

#         h_pair = self.conf_pair_proj(pair)   # [B, N, N, H*C]
#         h_pair = F.leaky_relu(h_pair, negative_slope=0.2)
#         h_pair = h_pair.view(B, N, N, self.num_heads, self.head_dim)

#         # attention scores
#         # [B, N, N, H]
#         scores = (h_pair * self.conf_att.view(1, 1, 1, self.num_heads, self.head_dim)).sum(dim=-1)

#         # optionally incorporate edge strength if adj is weighted
#         # safer than hard replacing with huge constants
#         if adj_fp32.max() > 1.0 or (adj_fp32.min() < 0.0):
#             # do nothing special if weird weights
#             pass
#         else:
#             # bias by log(weight) when weights are in [0, 1+] and >0
#             edge_bias = torch.zeros_like(adj_fp32)
#             positive = adj_fp32 > 0
#             edge_bias[positive] = torch.log(adj_fp32[positive] + 1e-6)
#             scores = scores + edge_bias.unsqueeze(-1)

#         # mask
#         scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))

#         # softmax over neighbor j
#         attn = torch.softmax(scores, dim=2)
#         attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

#         # if no neighbors, set whole row to zero
#         attn = torch.where(
#             has_nbr.unsqueeze(-1),
#             attn,
#             torch.zeros_like(attn)
#         )

#         # apply adjacency strength again and renormalize
#         attn = attn * adj_fp32.unsqueeze(-1)
#         denom = attn.sum(dim=2, keepdim=True) + 1e-8
#         attn = attn / denom

#         # values
#         v = self.conf_v(x_fp32).view(B, N, self.num_heads, self.head_dim)  # [B, N, H, C]
#         v = v.unsqueeze(1)  # [B, 1, N, H, C]

#         # aggregate over j
#         out = (attn.unsqueeze(-1) * v).sum(dim=2)  # [B, N, H, C]
#         out = out.reshape(B, N, self.num_heads * self.head_dim)
#         out = self.conf_out(out).to(dtype)

#         return out

#     def forward(self, x: torch.Tensor, conflict_index: torch.Tensor, same_index: torch.Tensor):
#         """
#         x:              [B, N, D]
#         conflict_index: [B, N, N]
#         same_index:     [B, N, N]
#         """
#         B, N, D = x.shape
#         device = x.device

#         I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

#         # --------------------------------------------------
#         # 1) SAME branch: JK-style multi-hop
#         # --------------------------------------------------
#         A_same = symmetric_normalize(same_index + I).to(x.dtype)

#         h0 = F.relu(self.same_fc0(x))              # [B, N, D]
#         h1 = torch.bmm(A_same, F.relu(self.same_fc1(x)))
#         h2 = torch.bmm(A_same, torch.bmm(A_same, F.relu(self.same_fc2(x))))

#         jk_logits = self.same_jk_gate(torch.cat([h0, h1, h2], dim=-1))   # [B, N, 3]
#         jk_alpha = torch.softmax(jk_logits, dim=-1)

#         msg_same = (
#             jk_alpha[..., 0:1] * h0 +
#             jk_alpha[..., 1:2] * h1 +
#             jk_alpha[..., 2:3] * h2
#         )
#         msg_same = self.same_out(msg_same)
#         h_same = self.norm_same(x + msg_same)

#         # --------------------------------------------------
#         # 2) CONFLICT branch: GATv2-style attention
#         # --------------------------------------------------
#         conf_nbr = self.conflict_gatv2_attention(x, conflict_index)

#         delta_conf = x - conf_nbr
#         prod_conf = x * conf_nbr

#         msg_conf = self.conf_mlp(torch.cat([x, conf_nbr, delta_conf, prod_conf], dim=-1))
#         h_conf = self.norm_conf(x + msg_conf)

#         # --------------------------------------------------
#         # 3) FUSION
#         # --------------------------------------------------
#         gates = self.fuse(torch.cat([x, h_same, h_conf], dim=-1))
#         alpha, beta = torch.chunk(gates, 2, dim=-1)
#         alpha = torch.sigmoid(alpha)
#         beta = torch.sigmoid(beta)

#         out = self.norm_out(x + alpha * msg_same + beta * msg_conf)
#         return out