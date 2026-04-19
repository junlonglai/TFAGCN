import numpy as np


def get_pos_emb(pos, hid_dim):
    pos_emb = np.zeros((1, hid_dim))
    for i in range(hid_dim):
        if i % 2 == 0:
            pos_emb[0, i] = np.sin(pos/(10000**(i/hid_dim)))
        else:
            pos_emb[0, i] = np.cos(pos/(10000**((i-1)/hid_dim)))
    return pos_emb

