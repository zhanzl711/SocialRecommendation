from torch import nn
import torch.nn.functional as F
import torch


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_rate_levels, emb_dim):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.user_emb_mlp = nn.Embedding(self.num_users, self.emb_dim, padding_idx=0)
        self.item_emb_mlp = nn.Embedding(self.num_items, self.emb_dim, padding_idx=0)
        self.user_emb_mf = nn.Embedding(self.num_users, self.emb_dim, padding_idx=0)
        self.item_emb_mf = nn.Embedding(self.num_items, self.emb_dim, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, 4 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(4 * self.emb_dim, 2 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        self.affine_output = nn.Linear(in_features=2 * emb_dim, out_features=1)

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad):
        u_emb_mlp = self.user_emb_mlp(uids)
        v_emb_mlp = self.item_emb_mlp(iids)
        u_emb_mf = self.user_emb_mf(uids)
        v_emb_mf = self.item_emb_mf(iids)

        mlp_vector = torch.cat([u_emb_mlp, v_emb_mlp], dim=-1)
        mf_vector = torch.mul(u_emb_mf, v_emb_mf)

        mlp_vector = self.mlp(mlp_vector)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        return logits




