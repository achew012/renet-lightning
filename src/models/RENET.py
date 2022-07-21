import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from models.Aggregator import RGCNAggregator
from collections import defaultdict
from common.utils import *
import pytorch_lightning as pl


class RENet(pl.LightningModule):
    r"""The Recurrent Event Network model from the `"Recurrent Event Network
    Args:
        num_nodes (int): The number of nodes in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        hidden_channels (int): Hidden size of node and relation embeddings.
        seq_len (int): The sequence length of past events.
        num_layers (int, optional): The number of recurrent layers.
            (default: :obj:`1`)
        dropout (float): If non-zero, introduces a dropout layer before the
            final prediction. (default: :obj:`0.`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """

    def __init__(self, conf):
        super().__init__()

        self.num_nodes = conf['num_nodes']
        self.h_dim = conf['n_hidden']
        self.num_rels = conf['num_rels']
        self.seq_len = conf['seq_len']
        self.conf = conf

        self.rel_embeds = nn.Parameter(
            torch.Tensor(2 * self.num_rels, self.h_dim))
        nn.init.xavier_uniform_(self.rel_embeds,
                                gain=nn.init.calculate_gain('relu'))
        self.ent_embeds = nn.Parameter(
            torch.Tensor(self.num_nodes, self.h_dim))
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(conf['dropout'])

        self.encoder = nn.GRU(3 * self.h_dim, self.h_dim, batch_first=True)

        self.aggregator = RGCNAggregator(
            self.h_dim, conf['dropout'], self.num_nodes, self.num_rels, conf['RGCN_bases']).to(self.device)

        self.linear = nn.Linear(3 * self.h_dim, self.num_nodes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch_data, subject=True, return_prob=False):

        if subject:
            s, r, o, len_non_zero, batched_graph, node_ids_tensor = batch_data
            rel_embeds = self.rel_embeds[:self.num_rels]
            reverse = False

        else:
            o, r, s, len_non_zero, batched_graph, node_ids_tensor = batch_data
            rel_embeds = self.rel_embeds[self.num_rels:]
            reverse = True

        s_packed_input = self.aggregator(len_non_zero, s, r, batched_graph, node_ids_tensor,
                                         self.ent_embeds, rel_embeds, reverse=reverse)

        if s_packed_input is not None:
            _, s_h = self.encoder(s_packed_input)
            s_h = s_h.squeeze(dim=0)
            s_h = torch.cat(
                (s_h, torch.zeros(len(s) - len(s_h), self.h_dim, device=self.device)), dim=0)
        else:
            s_h = torch.zeros(len(s), self.h_dim, device=self.device)

        final_rep = torch.cat((self.ent_embeds[s], s_h, rel_embeds[r]), dim=1)

        contrastive_loss = 0
        if self.conf['use_contrastive']:
            contrastive_loss = self.cal_infoNCE(final_rep)

        ob_pred = self.linear(self.dropout(final_rep))

        loss = self.criterion(ob_pred, o)

        if return_prob:
            return ob_pred, o

        return loss, contrastive_loss

    def cal_infoNCE(self, rep):
        # rep: [bs, hid_dim]
        rep = F.normalize(rep, p=2, dim=1)
        pos_score = torch.sum(rep * rep, dim=1)  # [bs]
        ttl_score = torch.matmul(rep, rep.permute(1, 0))  # [bs, bs]

        pos_score = torch.exp(pos_score / self.conf["c_temp"])  # [bs]
        ttl_score = torch.sum(
            torch.exp(ttl_score / self.conf["c_temp"]), dim=1)  # [bs]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def test(self, logits, y):
        """Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10."""

        _, perm = logits.sort(dim=1, descending=True)
        mask = (y.view(-1, 1) == perm)

        nnz = mask.nonzero(as_tuple=False)
        mrr = (1 / (nnz[:, -1] + 1).to(torch.float)).mean().item()
        hits1 = mask[:, :1].sum().item() / y.size(0)
        hits3 = mask[:, :3].sum().item() / y.size(0)
        hits10 = mask[:, :10].sum().item() / y.size(0)

        return torch.tensor([mrr, hits1, hits3, hits10])
