import dgl
import random
import itertools
import pandas as pd
import argparse
import configparser
import dgl.sparse as dglsp
from lib.metrics import *
from model.TFAGS import TFAGS, DotPredictor
from model.loss import get_TFAGS_loss


# ==============
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='wiki', type=str)
args = parser.parse_args()

# ==============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# ==============
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(28)

# ==============
## 读取数据
data_name = args.dataset
edges_data = pd.read_csv('./data/%s.txt' % data_name, sep=' ', header=None, names=['from', 'to', 'time'])
num_nodes = 1140149

# ==============
## 常规配置
config_path = 'configurations/' + data_name + '.conf'
config = configparser.ConfigParser()
config.read(config_path)
training_config = config['Training']
epsilon = float(training_config['epsilon'])  # 置零的阈值
num_epochs = int(training_config['num_epochs'])  # 训练代数
win_size = 10  # 输入的快照数量
num_snaps = 26
num_samples = num_snaps - win_size  # 样本数
num_test_snaps = int(num_samples * 0.2)  # 测试样本数
num_val_snaps = int(num_samples * 0.1)  # 验证样本数
num_train_snaps = num_samples-num_test_snaps-num_val_snaps  # 训练样本数

# ====================
## 定义模型
model = TFAGS(num_nodes, win_size).to(device)
pred = DotPredictor().to(device)

# ==========
## 定义优化器
opt = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=5e-4, weight_decay=1e-5)

# ====================
## 训练过程
rand_proj = torch.randn(num_nodes, 32)
embedding = torch.nn.Embedding(num_nodes, 16).to(device)
for epoch in range(num_epochs):
    model.train()
    loss_list = []
    for k in range(1, num_train_snaps + 1):
        G_list = []
        subgraph_adjs = []
        for t in range(k, k + win_size):
            group = edges_data[edges_data['time'] == t]
            edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
            edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            # graph = dgl.to_simple(graph)
            # graph = dgl.add_reverse_edges(graph)
            g = dgl.to_bidirected(graph)
            G_list.append(g.to(device))

            adj_emb = dglsp.bspmm(g.adjacency_matrix(), rand_proj)
            subgraph_adjs.append(adj_emb)
        subgraph_adjs = torch.stack(subgraph_adjs).to(device)

        ### ==========
        nodes_emb = model(G_list, subgraph_adjs, embedding(G_list[0].nodes()))

        group = edges_data[edges_data['time'] == (k+win_size)]  # Label
        edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
        edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
        graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
        G_gnd_pos = dgl.to_bidirected(graph).to(device)
        neg_nums = G_gnd_pos.number_of_edges()
        G_gnd_neg = dgl.graph(dgl.sampling.global_uniform_negative_sampling(G_gnd_pos, neg_nums), num_nodes=num_nodes).to(device)

        pos_score = pred(G_gnd_pos, nodes_emb).to(device)
        neg_score = pred(G_gnd_neg, nodes_emb).to(device)

        loss = get_TFAGS_loss(pos_score, neg_score, device)
        ### ==========
        opt.zero_grad()
        loss.backward()
        opt.step()
        ### ==========
        loss_list.append(loss.item())

    ### ==========
    loss_mean = np.mean(loss_list)
    print('Epoch %d Train Loss %f' % (epoch, loss_mean))

    # ====================
    # 验证过程
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        for k in range(num_train_snaps + 1, num_train_snaps + num_val_snaps + 1):
            G_list = []
            subgraph_adjs = []
            for t in range(k, k + win_size):
                group = edges_data[edges_data['time'] == t]
                edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
                edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
                graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
                g = dgl.to_bidirected(graph)
                G_list.append(g.to(device))

                adj_emb = dglsp.bspmm(g.adjacency_matrix(), rand_proj)
                subgraph_adjs.append(adj_emb)
            subgraph_adjs = torch.stack(subgraph_adjs).to(device)

            ### ==========
            nodes_emb = model(G_list, subgraph_adjs, embedding(G_list[0].nodes()))

            group = edges_data[edges_data['time'] == (k + win_size)]  # Label
            edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
            edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            G_gnd_pos = dgl.to_bidirected(graph).to(device)
            neg_nums = G_gnd_pos.number_of_edges()
            G_gnd_neg = dgl.graph(dgl.sampling.global_uniform_negative_sampling(G_gnd_pos, neg_nums),
                                  num_nodes=num_nodes).to(device)

            pos_score = pred(G_gnd_pos, nodes_emb).to(device)
            neg_score = pred(G_gnd_neg, nodes_emb).to(device)

            val_loss = get_TFAGS_loss(pos_score, neg_score, device)
            val_loss_list.append(val_loss.item())

        ### ==========
        if len(val_loss_list) > 1:
            val_loss_mean = np.mean(val_loss_list)
        else:
            val_loss_mean = val_loss_list[0]
        print('Epoch %d Val Loss %f' % (epoch, val_loss_mean))

    # ====================
    # 测试过程
    model.eval()
    with torch.no_grad():
        AUC_list = []
        Recall_list = []
        for k in range(num_samples - num_test_snaps + 1, num_samples + 1):
            G_list = []
            subgraph_adjs = []
            for t in range(k, k + win_size):
                group = edges_data[edges_data['time'] == t]
                edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
                edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
                graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
                g = dgl.to_bidirected(graph)
                G_list.append(g.to(device))

                adj_emb = dglsp.bspmm(g.adjacency_matrix(), rand_proj)
                subgraph_adjs.append(adj_emb)
            subgraph_adjs = torch.stack(subgraph_adjs).to(device)

            ### ==========
            nodes_emb = model(G_list, subgraph_adjs, embedding(G_list[0].nodes()))

            group = edges_data[edges_data['time'] == (k + win_size)]  # Label
            edges_src = torch.from_numpy(group['from'].to_numpy(dtype=np.int32))
            edges_dst = torch.from_numpy(group['to'].to_numpy(dtype=np.int32))
            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
            G_gnd_pos = dgl.to_bidirected(graph).to(device)
            neg_nums = G_gnd_pos.number_of_edges()
            G_gnd_neg = dgl.graph(dgl.sampling.global_uniform_negative_sampling(G_gnd_pos, neg_nums),
                                  num_nodes=num_nodes).to(device)

            pos_score = pred(G_gnd_pos, nodes_emb).to(device)
            neg_score = pred(G_gnd_neg, nodes_emb).to(device)

            auc = compute_auc(pos_score, neg_score)
            recall = compute_recall(pos_score, neg_score, epsilon)
            AUC_list.append(auc)
            Recall_list.append(recall)
        ### ==========
        AUC_mean = np.mean(AUC_list)
        Recall_mean = np.mean(Recall_list)
        print('Epoch %d Test AUC %f Recall %f' % (epoch, AUC_mean, Recall_mean))
