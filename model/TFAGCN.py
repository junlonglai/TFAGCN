import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as Init
from torch.nn.parameter import Parameter


class TFAGCN(nn.Module):
    def __init__(self, GT_dims, ST_dims, num_nodes, dropout_rate):
        super(TFAGCN, self).__init__()
        self.GTL = Global_temporal_layer(GT_dims[0], num_nodes)
        self.STL = Spatio_temporal_layer(ST_dims[0], ST_dims[1], ST_dims[2], num_nodes, dropout_rate)

    def forward(self, adj_tnr_list, pos_tnr):
        # ==========
        ## Global_temporal_layer
        fre_feat = self.GTL(adj_tnr_list)
        # ==========
        ## Spatio_temporal_layer
        adj_est = self.STL(adj_tnr_list, fre_feat, pos_tnr)

        return adj_est


class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, dropout_rate):
        super(GCN, self).__init__()
        self.W_agg = Init.xavier_uniform_(Parameter(torch.FloatTensor(in_feat, out_feat)))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, feat, adj):
        A_hat = adj + torch.eye(adj.shape[0]).type_as(adj)
        D = torch.diag(torch.sum(A_hat, dim=1))
        D_sqrt_inv = torch.inverse(torch.sqrt(D))
        A_hat_norm = torch.matmul(torch.matmul(D_sqrt_inv, A_hat), D_sqrt_inv)

        feat_agg = torch.spmm(A_hat_norm, feat)  # (N,in_feat)
        agg_output = torch.relu(torch.matmul(feat_agg, self.W_agg))  # (N,in_feat)(in_feat,out_feat)->(N,out_feat)
        norm_output = F.normalize(agg_output, dim=1, p=2)  # l2-normalization
        gcn_output = self.dropout(norm_output)  # (N,out_feat)

        return gcn_output


class Time_fre_attention(nn.Module):
    def __init__(self, d_in, d_model, dropout_rate):
        super(Time_fre_attention, self).__init__()
        self.d_model = d_model
        self.q0_linear = nn.Linear(d_in, d_model)
        self.k0_linear = nn.Linear(d_in, d_model)
        self.v0_linear = nn.Linear(d_in, d_model)

        self.q1_linear = nn.Linear(d_in, d_model)
        self.k1_linear = nn.Linear(d_in, d_model)
        self.v1_linear = nn.Linear(d_in, d_model)

        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, feat_0, feat_1):
        # ==========
        ## 将输入线性转换为查询(q)、键(k)和值(v)
        q0 = self.q0_linear(feat_0)  # (N,d_model)
        k0 = self.k0_linear(feat_0)  # (N,d_model)
        v0 = self.v0_linear(feat_0)  # (N,d_model)

        q1 = self.q1_linear(feat_1)  # (N,d_model)
        k1 = self.k1_linear(feat_1)  # (N,d_model)
        v1 = self.v1_linear(feat_1)  # (N,d_model)
        # ==========
        ## 计算注意力分数
        s0 = torch.matmul(q1, k0.transpose(-2, -1)) / (self.d_model ** 0.5)  # (N,N)
        s1 = torch.matmul(q0, k1.transpose(-2, -1)) / (self.d_model ** 0.5)  # (N,N)
        ## 使用softmax来获得注意力权重
        attn_w0 = F.softmax(s0, dim=-1)  # (N,N)
        attn_w1 = F.softmax(s1, dim=-1)  # (N,N)
        # ==========
        ## 使用注意力权重对值向量进行加权求和
        context_0 = torch.matmul(attn_w0, v0)  # (N,N)(N,d_model)->(N,d_model)
        context_1 = torch.matmul(attn_w1, v1)  # (N,N)(N,d_model)->(N,d_model)
        # ==========
        ## 特征相加
        context = context_0 + context_1  # (N,d_model)
        # output = self.dropout(context)  # (N,d_model)

        return context


class Global_temporal_layer(nn.Module):
    def __init__(self, embed_size, num_nodes):
        super(Global_temporal_layer, self).__init__()
        self.W_emb = Init.xavier_uniform_(Parameter(torch.FloatTensor(num_nodes, embed_size)))
        self.b_emb = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, embed_size)))
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.r = Init.xavier_uniform_(Parameter(torch.FloatTensor(embed_size, embed_size)))
        self.i = Init.xavier_uniform_(Parameter(torch.FloatTensor(embed_size, embed_size)))
        self.rb = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, embed_size)))
        self.ib = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, embed_size)))

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
        y = F.softshrink(y, lambd=0.01)  # 软阈值稀疏化,小于阈值置0，否则不变
        y = torch.view_as_complex(y)
        return y

    def forward(self, adj_tnr_list):
        # ==========
        ## embeddings
        adj_tnr_seq = torch.stack(adj_tnr_list)   # (T,N,N)
        adj_emb_seq = torch.einsum('tij, jd -> tid', adj_tnr_seq, self.W_emb) + self.b_emb  # (T,N,D)
        # adj_emb_seq = self.dropout(adj_emb_seq)
        adj_emb_seq = torch.relu(adj_emb_seq)
        x = adj_emb_seq.permute(1, 0, 2)  # (T,N,D)->(N,T,D)
        # ==========
        ## frequency temporal learner
        bias = x
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # FFT on T dimension: (N,T,D)->(N,T/2+1,D)
        y = self.FreMLP(x, self.r, self.i, self.rb, self.ib)
        x = torch.fft.irfft(y, n=len(adj_tnr_list), dim=1, norm="ortho")  # IFFT on T dimension: (N,T/2+1,D)->(N,T,D)
        x = x + bias  # (N,T,D)

        return x


class Spatio_temporal_layer(nn.Module):
    def __init__(self, pos_dim, h_dim, mid_dim, num_nodes, dropout_rate):
        super(Spatio_temporal_layer, self).__init__()
        # ==========
        # GNN
        self.gcn = GCN(pos_dim, pos_dim, dropout_rate)
        # ==========
        # GRU
        self.h_dim = h_dim
        ## Reset gate parameters
        self.Wr = Init.xavier_uniform_(Parameter(torch.FloatTensor(pos_dim + h_dim, h_dim)))
        self.br = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, h_dim)))
        ## Update gate parameters
        self.Wz = Init.xavier_uniform_(Parameter(torch.FloatTensor(pos_dim + h_dim, h_dim)))
        self.bz = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, h_dim)))
        ## Candidate update parameters
        self.W = Init.xavier_uniform_(Parameter(torch.FloatTensor(pos_dim + h_dim, h_dim)))
        self.b = Init.xavier_uniform_(Parameter(torch.FloatTensor(1, h_dim)))
        # ==========
        # Time-frequency bidirectional cross-attention
        self.cross_att = Time_fre_attention(h_dim, h_dim, dropout_rate)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, mid_dim),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(mid_dim, num_nodes),
            nn.Sigmoid()
        )

    def forward(self, adj_list, fre_feat, pos_feat):
        # ==========
        ## GNN & GRU
        N, T, fre_dim = fre_feat.shape  # (N,T,D)
        st_feat = list()
        h_prev = torch.randn(N, self.h_dim).type_as(fre_feat)  # (N,h_dim)
        for t in range(len(adj_list)):
            spatial_feat = self.gcn(pos_feat, adj_list[t])  # (N,pos_dim)
            concat_input = torch.cat((spatial_feat, h_prev), dim=1)  # (N,h_dim+pos_dim)
            r = torch.sigmoid(torch.matmul(concat_input, self.Wr) + self.br)  # Reset gate
            z = torch.sigmoid(torch.matmul(concat_input, self.Wz) + self.bz)  # Update gate
            h_tilde = torch.tanh(torch.matmul(torch.cat((spatial_feat, r * h_prev), dim=1), self.W) + self.b)  # Candidate update
            h = (1 - z) * h_prev + z * h_tilde  # Update hidden state
            h_prev = self.cross_att(h, fre_feat[:, t, :])
            st_feat.append(h_prev)
        last_output = st_feat[-1]  # (N,h_dim)
        adj_est = self.mlp(last_output)  # (N,N)

        return adj_est
