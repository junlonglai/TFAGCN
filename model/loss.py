import torch
import torch.nn.functional as F


def get_TFAGCN_loss(adj_est, gnd, beta):
    p = torch.ones_like(gnd)
    p_beta = beta*p
    p = torch.where(gnd == 0, p, p_beta)
    loss = torch.norm(torch.mul((adj_est - gnd), p), p='fro') ** 2

    return loss

def get_TFAGS_loss(pos_score, neg_score, device=torch.torch.device('cpu')):

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

    return F.binary_cross_entropy_with_logits(scores, labels)
