from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, LayerNorm
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

import pdb


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin1 = Linear(hidden_channels, num_filters, bias=False)
        self.lin2 = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index, v1_size):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)

        v1 = self.lin1(v[:v1_size])
        v2 = self.lin2(v[v1_size:])
        v = torch.cat([v1, v2], dim=0)

        e = v[j] * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1_1 = Linear(num_filters, hidden_channels)
        self.lin1_2 = Linear(hidden_channels, hidden_channels)
        self.lin2_1 = Linear(num_filters, hidden_channels)
        self.lin2_2 = Linear(hidden_channels, hidden_channels)
        self.layer_norm = LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1_1.weight)
        self.lin1_1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin1_2.weight)
        self.lin1_2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2_1.weight)
        self.lin2_1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2_2.weight)
        self.lin2_2.bias.data.fill_(0)

    def forward(self, v, e, edge_index, v1_size):
        _, i = edge_index
        out = scatter(e, i, dim=0)

        out1 = out[:v1_size]
        out1 = self.lin1_1(out1)
        out1 = self.act(out1)
        out1 = self.lin1_2(out1)

        out2 = out[v1_size:]
        out2 = self.lin1_1(out2)
        out2 = self.act(out2)
        out2 = self.lin1_2(out2)

        out = torch.cat([out1, out2], dim=0)
        out = self.layer_norm(out)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.layer_norm = LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        u = scatter(v, batch, dim=0)
        u = self.layer_norm(u)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class HeSchNet(torch.nn.Module):
    r"""
        The re-implementation for HeSchNet from the `"HeSchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Output embedding size. (default: :obj:`1`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """
    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50):
        super(HeSchNet, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(18, hidden_channels)
        self.init_m_v = Embedding(25, hidden_channels)

        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])
        self.update_mvs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        for update_mv in self.update_mvs:
            update_mv.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, data):
        g = data.g

        z1, pos1, batch1 = data['ap'].nid,  data['ap'].pos, data['ap'].batch
        z2, pos2, batch2 = data['module'].midx,  data['module'].pos, data['module'].batch

        pos1 *= g
        # value_g = g.detach()
        pos2 *= g.detach()[0]
        
        pos = torch.cat([pos1, pos2], dim=0)
        batch = torch.cat([batch1, batch2], dim=0)

        if self.energy_and_force:
            g.requires_grad_()

        ##################
        # distance graph
        ##################
        # cost_pos = pos * g

        # edge_index = radius_graph(pos * g, r=self.cutoff, batch=batch)
        # edge_index = radius_graph(cost_pos, r=self.cutoff, batch=batch)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        # ap distance graph & ap module distance graph
        mask_1 = (edge_index[0] < len(pos1)) & (edge_index[1] < len(pos1))
        mask_2 = (edge_index[0] < len(pos1)) & (edge_index[1] >= len(pos1))

        edge_index = edge_index[:, mask_1 | mask_2]
        edge_index = torch.cat([edge_index, data['ap', 'connect','ap'].edge_index,
                                        data['ap', 'connect', 'module'].edge_index], dim=-1)

        row, col = edge_index
        # dist = ((pos[row] - pos[col]) * g).norm(dim=-1)
        # dist = (cost_pos[row] - cost_pos[col]).norm(dim=-1)

        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v1 = self.init_v(z1)
        pin_offset = torch.cos(PI / 2 + PI * (data['ap'].pid / data['ap'].pnum)).unsqueeze(-1)
        v1 += pin_offset

        v2 = self.init_m_v(z2)
        v = torch.concat([v1, v2], dim=0)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index, v1_size=len(pos1))
            v = update_v(v, e, edge_index, v1_size=len(pos1))

        # agression
        u = self.update_u(v, batch)
        return u

        # edge_index_1 = edge_index[:, mask_1]
        # edge_index_2 = edge_index[:, mask_2]

        # row, col = edge_index_1
        # dist1 = ((pos[row] - pos[col]) * g).norm(dim=-1)
        # dist1_emb = self.dist_emb(dist1)

        # row, col = edge_index_2
        # dist2 = ((pos[row] - pos[col]) * g).norm(dim=-1)
        # dist2_emb = self.dist_emb(dist2)
        # pdb.set_trace()

        # # connection graph
        # row_c, col_c = data['ap', 'connect','ap'].edge_index
        # dist_c = ((pos1[row_c] - pos1[col_c]) * g).norm(dim=-1)
        # dist_emb_c = self.dist_emb(dist_c)

        # row_p, col_p = data['ap', 'connect', 'module'].edge_index
        # dist_p = ((pos1[row_p] - pos2[col_p]) * g).norm(dim=-1)
        # dist_emb_p = self.dist_emb(dist_p)

        # for update_e, update_v, update_mv in zip(self.update_es, self.update_vs, self.update_mvs):
        #     ##################
        #     # update access points
        #     ##################

        #     pdb.set_trace()

        #     # distance graph
        #     e1 = update_e(v1, dist1, dist1_emb, edge_index_1)
        #     v1 = update_v(v1, e1, edge_index_1)

        #     # connection graph
        #     e_c = update_e(v1, dist_c, dist_emb_c, data['ap', 'connect','ap'].edge_index)
        #     v1 = update_v(v1, e_c, data['ap', 'connect','ap'].edge_index)

        #     pdb.set_trace()

        #     ##################
        #     # update access points and modules
        #     ##################

        #     # distance graph
        #     e2 = update_e(v, dist2, dist2_emb, edge_index_2)
        #     v = update_mv(v, e2, edge_index_2)

        #     # connection graph
        #     e_p = update_e(v, dist_p, dist_emb_p, data['ap', 'connect', 'module'].edge_index)
        #     v = update_mv(v, e_p, data['ap', 'connect', 'module'].edge_index)
        #     v2 = v[len(pos1):]

        #     v = torch.cat([v1, v2], dim=0)

        # pdb.set_trace()

        # v1 = self.init_v(z1)
        # for update_e, update_v in zip(self.update_es, self.update_vs):
        #     # distance graph
        #     e1 = update_e(v1, dist1, dist1_emb, edge_index_1)
        #     v1 = update_v(v1, e1, edge_index_1)

        #     # connection graph
        #     e_c = update_e(v1, dist_c, dist_emb_c, data['ap', 'connect','ap'].edge_index)
        #     v1 = update_v(v1, e_c, data['ap', 'connect','ap'].edge_index)

        # pdb.set_trace()
        # v2 = self.init_m_v(z2)
        # for update_e, update_mv in zip(self.update_es, self.update_mvs):
        #     # distance graph
        #     e2 = update_e(v2, dist2, dist2_emb, edge_index_2)
        #     v2 = update_mv(v2, e2, edge_index_2)

        #     # connection graph
        #     e_p = update_e(v2, dist_p, dist_emb_p, data['ap', 'connect', 'module'].edge_index)
        #     v2 = update_mv(v2, e_p, data['ap', 'connect', 'module'].edge_index)

        # pdb.set_trace()
        # # concat nodes
        # v = torch.concat([v1, v2], dim=0)

