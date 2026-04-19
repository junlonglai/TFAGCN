import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import recall_score, roc_auc_score


def get_RMSE(adj_est, gnd, num_nodes):
    f_norm = np.linalg.norm(gnd-adj_est, ord='fro')**2
    RMSE = np.sqrt(f_norm/(num_nodes*num_nodes))

    return RMSE

def get_AUC(adj_est, gnd, num_nodes):
    gnd_vec = np.reshape(gnd, [num_nodes*num_nodes])
    pred_vec = np.reshape(adj_est, [num_nodes*num_nodes])

    fpr, tpr, _ = metrics.roc_curve(gnd_vec, pred_vec)
    AUC = metrics.auc(fpr, tpr)

    return AUC

def get_MR(adj_est, gnd, num_nodes):
    num_mismatches = np.sum((adj_est > 0) != (gnd > 0))
    MR = num_mismatches / (num_nodes * num_nodes)

    return MR

def get_Recall(adj_est, gnd, num_nodes):
    adj_est[adj_est != 0] = 1
    gnd_vec = np.reshape(gnd, [num_nodes*num_nodes])
    pred_vec = np.reshape(adj_est, [num_nodes*num_nodes])
    recall = recall_score(gnd_vec, pred_vec)

    return recall

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()

    return roc_auc_score(labels, scores)

def compute_recall(pos_score, neg_score, threshold=0.05):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    pred_labels = (scores >= threshold).astype(int)
    recall = recall_score(labels, pred_labels)

    return recall
