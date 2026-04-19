import torch.nn.functional as F
import torch
import torch.nn as nn
from dgl.nn import SAGEConv
import dgl.function as fn
import torch.nn.init as Init
from torch.nn.parameter import Parameter

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class TFAGS(nn.Module):
    def __init__(self, num_nodes, win_size):
        super(TFAGS, self).__init__()
        self.num_nodes = num_nodes
        self.win_size = win_size

        self.r = Init.xavier_uniform_(Parameter(torch.FloatTensor(32, 32)))
        self.i = Init.xavier_uniform_(Parameter(torch.FloatTensor(32, 32)))
        self.rb = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, 32)))
        self.ib = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, 32)))

        self.conv1 = SAGEConv(16, 16, 'mean')
        self.r_linear = nn.Linear(16+32, 32)
        self.z_linear = nn.Linear(16+32, 32)
        self.h_linear = nn.Linear(16+32, 32)
        self.ft_linear = nn.Linear(2 * 32, 32)

    # FreMLP
    def FreMLP(self, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum('ntd, dd->ntd', x.real, r) - \
            torch.einsum('ntd, dd->ntd', x.imag, i) + \
            rb
        )
        o1_imag = F.relu(
            torch.einsum('ntd, dd->ntd', x.imag, r) + \
            torch.einsum('ntd, dd->ntd', x.real, i) + \
            ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=0.01)
        y = torch.view_as_complex(y)
        return y

    def forward(self, G_list, subgraph_adjs, embedding):
        # ==========
        ## Global_temporal_layer
        x = subgraph_adjs.permute(1, 0, 2)  # (T,N,D)->(N,T,D)
        bias = x
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # FFT on T dimension: (N,T,D)->(N,T/2+1,D)
        y = self.FreMLP(x, self.r, self.i, self.rb, self.ib)
        x = torch.fft.irfft(y, n=self.win_size, dim=1, norm="ortho")  # IFFT on T dimension: (N,T/2+1,D)->(N,T,D)
        fre_feat = x + bias  # (N,T,D)

        # ==========
        ## Spatio_temporal_layer
        in_feat = embedding
        h_prev = torch.randn(self.num_nodes, 32).type_as(fre_feat)  # (N,h_dim)
        for t in range(self.win_size):
            g = G_list[t]
            h = self.conv1(g, in_feat)
            spatial_feat = F.relu(h)  # (N,pos_dim)
            concat_input = torch.cat((spatial_feat, h_prev), dim=1)  # (N,h_dim+pos_dim)
            r = torch.sigmoid(self.r_linear(concat_input))  # Reset gate
            z = torch.sigmoid(self.z_linear(concat_input))  # Update gate
            h_tilde = torch.tanh(self.h_linear(torch.cat((spatial_feat, r * h_prev), dim=1)))  # Candidate update
            h = (1 - z) * h_prev + z * h_tilde  # Update hidden state
            h_prev = self.ft_linear(torch.cat((h, fre_feat[:, t, :]), dim=1))  # (N,d_model)
        nodes_emd = h_prev  # (N,h_dim)

        return nodes_emd
