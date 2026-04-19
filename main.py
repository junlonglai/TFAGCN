import os
import random
import argparse
import configparser
from lib.utils import *
from lib.metrics import *
import torch.optim as optim
from model.TFAGCN import TFAGCN
from model.loss import get_TFAGCN_loss


# ==============
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='Haggle', type=str)
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
adj_seq = np.load('data/%s.npy' % data_name, allow_pickle=True)  # T*N*N

num_snaps = adj_seq.shape[0]  # 快照数量
num_nodes = adj_seq.shape[1]  # 节点数量

# ==============
## 常规配置
config_path = 'configurations/' + data_name + '.conf'
config = configparser.ConfigParser()
config.read(config_path)
training_config = config['Training']
epsilon = float(training_config['epsilon'])  # 置零的阈值
num_epochs = int(training_config['num_epochs'])  # 训练代数
win_size = 10  # 输入的快照数量
beta = 2  # 损失函数惩罚系数
num_samples = num_snaps - win_size  # 样本数
num_test_snaps = int(num_samples * 0.2)  # 测试样本数
num_val_snaps = int(num_samples * 0.1)  # 验证样本数
num_train_snaps = num_samples-num_test_snaps-num_val_snaps  # 训练样本数

# ==============
## 模型参数配置
dropout_rate = 0
pos_dim = 16
GT_dims = [32]  # (embed_size)
ST_dims = [pos_dim, 32, 64]  # (pos_dim, h_dim, mid_dim)

# ==============
## 数据处理
### 原始矩阵
adj_tnr_seq = []
for t in range(num_snaps):
    adj = adj_seq[t, :, :]  # N*N
    adj_tnr = torch.FloatTensor(adj).to(device)
    adj_tnr_seq.append(adj_tnr)
### 位置嵌入
pos_emb = None
for p in range(num_nodes):
    if p == 0:
        pos_emb = get_pos_emb(p, pos_dim)
    else:
        pos_emb = np.concatenate((pos_emb, get_pos_emb(p, pos_dim)), axis=0)
pos_tnr = torch.FloatTensor(pos_emb).to(device)  # N*f

# ====================
## 定义模型
model = TFAGCN(GT_dims, ST_dims, num_nodes, dropout_rate).to(device)

# ==========
## 定义优化器
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# ====================
## 实验
# 检查文件夹是否存在，如果不存在则创建
params_path = os.path.join('exp')
params_filename = os.path.join(params_path, data_name + '_best_model.params')
if not os.path.exists(params_path):
    os.makedirs(params_path)

# ====================
## 训练过程
best_epoch = 0
best_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()
    loss_list = []
    for k in range(num_train_snaps):
        adj_tnr_list = adj_tnr_seq[k: k + win_size]
        gnd_tnr = adj_tnr_seq[k + win_size]  # N*N
        ### ==========
        adj_est = model(adj_tnr_list, pos_tnr)
        loss = get_TFAGCN_loss(adj_est, gnd_tnr, beta)
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
    val_loss_list = []
    for k in range(num_train_snaps, num_train_snaps + num_val_snaps):
        adj_tnr_list = adj_tnr_seq[k: k + win_size]
        gnd_tnr = adj_tnr_seq[k + win_size]  # N*N
        ### ==========
        adj_est = model(adj_tnr_list, pos_tnr)
        ### ==========
        #### 细化预测结果
        adj_est = (adj_est + adj_est.t()) / 2
        torch.diagonal(adj_est).fill_(0)
        adj_est[adj_est <= epsilon] = 0
        ### ==========
        val_loss = get_TFAGCN_loss(adj_est, gnd_tnr, beta)
        val_loss_list.append(val_loss.item())
    ### ==========
    val_loss_mean = np.mean(val_loss_list)
    print('Epoch %d Val Loss %f' % (epoch, val_loss_mean))
    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        best_epoch = epoch
        torch.save(model.state_dict(), params_filename)

    # ====================
    # 测试过程
    model.eval()
    AUC_list = []
    MR_list = []
    Recall_list = []
    for k in range(num_train_snaps + num_val_snaps, num_samples):
        adj_tnr_list = adj_tnr_seq[k: k + win_size]
        gnd = adj_seq[k + win_size]  # N*N
        ### ==========
        adj_est = model(adj_tnr_list, pos_tnr)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        ### ==========
        #### 细化预测结果
        adj_est = (adj_est + adj_est.T) / 2
        np.fill_diagonal(adj_est, 0)
        adj_est[adj_est <= epsilon] = 0
        ### ==========
        AUC = get_AUC(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        Recall = get_Recall(adj_est, gnd, num_nodes)
        ### ==========
        AUC_list.append(AUC)
        MR_list.append(MR)
        Recall_list.append(Recall)
    ### ==========
    AUC_mean = np.mean(AUC_list)
    MR_mean = np.mean(MR_list)
    Recall_mean = np.mean(Recall_list)
    print('Epoch %d Test AUC %f MR %f Recall %f'
          % (epoch, AUC_mean, MR_mean, Recall_mean))
    print()

# ====================
# 最佳模型测试过程
model.load_state_dict(torch.load(params_filename))
model.eval()
AUC_list = []
MR_list = []
Recall_list = []
for k in range(num_train_snaps + num_val_snaps, num_samples):
    adj_tnr_list = adj_tnr_seq[k: k + win_size]
    gnd = adj_seq[k + win_size]  # N*N
    ### ==========
    adj_est = model(adj_tnr_list, pos_tnr)
    if torch.cuda.is_available():
        adj_est = adj_est.cpu().data.numpy()
    else:
        adj_est = adj_est.data.numpy()
    ### ==========
    #### 细化预测结果
    adj_est = (adj_est + adj_est.T) / 2
    np.fill_diagonal(adj_est, 0)
    adj_est[adj_est <= epsilon] = 0
    ### ==========
    AUC = get_AUC(adj_est, gnd, num_nodes)
    MR = get_MR(adj_est, gnd, num_nodes)
    Recall = get_Recall(adj_est, gnd, num_nodes)
    ### ==========
    AUC_list.append(AUC)
    MR_list.append(MR)
    Recall_list.append(Recall)
### ==========
AUC_mean = np.mean(AUC_list)
MR_mean = np.mean(MR_list)
Recall_mean = np.mean(Recall_list)
f = open("exp/result.txt", 'a')
f.write(data_name + "\n")
f.write('AUC:{}, MR:{}, Recall:{}'.format(AUC_mean, MR_mean, Recall_mean))
f.write('\n')
f.write('\n')
f.close()
print('=================================================')
print('Best model: ')
print('Best_Epoch %d Test AUC %f MR %f Recall %f'
      % (best_epoch, AUC_mean, MR_mean, Recall_mean))

