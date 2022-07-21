import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from common.utils import *
from models.RGCN import RGCNBlockLayer as RGCNLayer
import pytorch_lightning as pl


class RGCNAggregator(pl.LightningModule):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases):
        super(RGCNAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.num_rels = num_rels
        self.num_nodes = num_nodes

        self.rgcn1 = RGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, num_bases,
                               activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, num_bases,
                               activation=None, self_loop=True, dropout=dropout)

    def forward(self, s_len_non_zero, s, r, batched_graph, node_ids_tensor, ent_embeds,
                rel_embeds, reverse):
        if len(s_len_non_zero) == 0:
            s_packed_input = None
        else:
            if batched_graph is None:
                s_packed_input = None
            else:
                batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']
                                                      ].view(-1, ent_embeds.shape[1])

                self.rgcn1(batched_graph, reverse)
                self.rgcn2(batched_graph, reverse)

                # [batched_graph.num_nodes, embedding_dim]
                embeds_mean = batched_graph.ndata.pop('h')
                embeds_mean = torch.cat(
                    [embeds_mean, torch.zeros(1, embeds_mean.shape[1]).cuda()], dim=0)

                s_repeat = s.unsqueeze(1).repeat(1, node_ids_tensor.shape[1])
                r_repeat = r.unsqueeze(1).repeat(1, node_ids_tensor.shape[1])
                s_embed_seq_tensor = torch.cat([embeds_mean[node_ids_tensor],
                                                ent_embeds[s_repeat],
                                                rel_embeds[r_repeat]],
                                               dim=-1)[:len(s_len_non_zero)]

                s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

                # s_embed_seq_tensor [bs, max_len_of_the_batch, embedding_dim]

                s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                         s_len_non_zero.cpu(),
                                                                         batch_first=True)

        return s_packed_input
