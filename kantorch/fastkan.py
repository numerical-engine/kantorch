import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SplineLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, init_scale:float = 0.1, **kwargs)->None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kwargs)
    def reset_parameters(self)->None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min:float = -2., grid_max:float = 2., num_grids:int = 8, denominator:float = None,)->None:
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKAN(nn.Module):
    def __init__(self,
        input_dim:int, output_dim:int, grid_min:float = -2., grid_max:float = 2., num_grids:int = 8, use_base_update:bool = True, use_layernorm:bool = True,
        base_activation = F.silu, spline_weight_init_scale: float = 0.1,)->None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        else:
            self.layernorm = None

        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class AttentionWithFastKANTransform(nn.Module):
    def __init__(self, q_dim:int, k_dim:int, v_dim:int, head_dim:int, num_heads:int, gating:bool = True,)->None:
        super().__init__()
        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        self.norm = head_dim**-0.5

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, bias:torch.Tensor = None,)->torch.Tensor:
        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)
        att = (wq * wk).sum(-1).softmax(-2)
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]
        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)
        o = (att[..., None] * wv).sum(-3)
        del att, wv
        o = o.view(*o.shape[:-2], -1)
        if self.linear_g is not None:
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o
        o = self.linear_o(o)
        return o