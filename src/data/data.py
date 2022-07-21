import json
import os
import pickle
from collections import defaultdict
from common import utils

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from clearml import StorageManager, Dataset as ClearMLDataset


def add_graph_id(graph_list_sorted):
    g_list = []
    for index, graph in enumerate(graph_list_sorted):
        if index == 0:
            graph.start_id = 0
        else:
            graph.start_id = graph_list_sorted[index - 1].start_id + \
                graph_list_sorted[index - 1].number_of_nodes()
        g_list.append(graph)

    return g_list


def get_node_ids_to_g_id(graph_list_sorted,
                         bs, max_len, node_id_list_sorted):
    node_ids_tensor = torch.neg(torch.ones(bs, max_len))

    for i, node_ids in enumerate(node_id_list_sorted):
        graph = graph_list_sorted[i]
        id_to_append = list(map(lambda x: x+graph.start_id, node_ids))
        node_ids_tensor[i][torch.arange(len(id_to_append))] = torch.tensor(
            id_to_append).float()

    return node_ids_tensor.long()


def custom_collate(data):
    s, r, o, window_batched_graph_s, window_batched_graph_o, node_id_list_s, node_id_list_o = zip(
        *data)

    bs = len(r)
    graph_list_sorted_s, graph_list_sorted_o = [], []
    node_id_list_sorted_s,  node_id_list_sorted_o = [], []

    s = torch.tensor(s)
    r = torch.tensor(r)
    o = torch.tensor(o)

    max_len_s = max(list(map(len, node_id_list_s)))
    s_hist_len = torch.LongTensor(list(map(len, node_id_list_s)))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    for i, idx in enumerate(s_idx):
        if i == num_non_zero:
            break
        graph_list_sorted_s.append(window_batched_graph_s[idx])
        node_id_list_sorted_s.append(node_id_list_s[idx])

    max_len_o = max(list(map(len, node_id_list_o)))
    o_hist_len = torch.LongTensor(list(map(len, node_id_list_o)))
    o_len, o_idx = o_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(o_len))
    o_len_non_zero = o_len[:num_non_zero]
    for i, idx in enumerate(o_idx):
        if i == num_non_zero:
            break
        graph_list_sorted_o.append(window_batched_graph_o[idx])
        node_id_list_sorted_o.append(node_id_list_o[idx])

    graph_list_sorted_s = add_graph_id(graph_list_sorted_s)
    graph_list_sorted_o = add_graph_id(graph_list_sorted_o)

    node_ids_tensor_s = get_node_ids_to_g_id(graph_list_sorted_s,
                                             bs, max_len_s, node_id_list_sorted_s)
    node_ids_tensor_o = get_node_ids_to_g_id(graph_list_sorted_o,
                                             bs, max_len_o, node_id_list_sorted_o)

    batched_graph_s = dgl.batch(
        graph_list_sorted_s) if graph_list_sorted_s else None
    batched_graph_o = dgl.batch(
        graph_list_sorted_o) if graph_list_sorted_o else None

    return {"s": [s[s_idx], r[s_idx], o[s_idx], s_len_non_zero, batched_graph_s, node_ids_tensor_s],
            "o": [s[o_idx], r[o_idx], o[o_idx], o_len_non_zero, batched_graph_o, node_ids_tensor_o]}


def test_collate(data):
    s, r, o, window_batched_graph, node_id_list = zip(*data)

    return s, r, o, window_batched_graph, node_id_list


class ComplexEventDataset(Dataset):
    def __init__(self, conf, data, graph_dict):
        self.data = data
        self.graph_dict = graph_dict
        self.conf = conf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        event_str = self.data[index]
        event = event_str.strip().split('\t')

        s, r, o, t, cid = [int(x) for x in event]

        g_list_s, g_list_o = [], []
        node_id_list_s, node_id_list_o = [], []

        if t > self.conf['seq_len']:
            history_range = range(t - self.conf['seq_len'], t)
        else:
            history_range = range(t)

        for tim in history_range:
            if s in self.graph_dict[cid][tim].ids:
                # don't need re-compute node degree norm even though leaf node norm is incorrect
                k_hop_subgraph, new_id = dgl.khop_out_subgraph(self.graph_dict[cid][tim],
                                                               self.graph_dict[cid][tim].ids[s],
                                                               k=2,
                                                               relabel_nodes=True)

                g_list_s.append(k_hop_subgraph)
                node_id_list_s.append(new_id.item())

            if o in self.graph_dict[cid][tim].ids:
                # don't need re-compute node degree norm even though leaf node norm is incorrect
                k_hop_subgraph, new_id = dgl.khop_out_subgraph(self.graph_dict[cid][tim],
                                                               self.graph_dict[cid][tim].ids[o],
                                                               k=2,
                                                               relabel_nodes=True)

                g_list_o.append(k_hop_subgraph)
                node_id_list_o.append(new_id.item())

        for index, graph in enumerate(g_list_s):
            if index == 0:
                graph.start_id = 0
            else:
                graph.start_id = g_list_s[index - 1].start_id + \
                    g_list_s[index - 1].number_of_nodes()
                node_id_list_s[index] += graph.start_id

        for index, graph in enumerate(g_list_o):
            if index == 0:
                graph.start_id = 0
            else:
                graph.start_id = g_list_o[index - 1].start_id + \
                    g_list_o[index - 1].number_of_nodes()
                node_id_list_o[index] += graph.start_id

        window_batched_graph_s = dgl.batch(g_list_s) if g_list_s else None
        window_batched_graph_o = dgl.batch(g_list_o) if g_list_o else None

        return s, r, o, window_batched_graph_s, window_batched_graph_o, node_id_list_s, node_id_list_o


class Datasets():
    def __init__(self, conf):
        if conf.remote:
            path = ClearMLDataset.get(
                dataset_project='datasets/gdelt_tkg', dataset_name='dense50EG_no_overlap').get_local_copy()
        else:
            path = conf['data_path']
        self.name = conf['dataset']
        # load data
        train_data, train_graph_dict, val_data, val_graph_dict, test_data, test_graph_dict = self.load_data(
            path)
        self.num_nodes, self.num_rels = utils.get_total_number(
            path, 'stat.txt')
        conf['num_nodes'], conf['num_rels'] = self.num_nodes, self.num_rels
        train_dataset = ComplexEventDataset(conf, train_data, train_graph_dict)
        self.train_loader = DataLoader(
            train_dataset, num_workers=4, collate_fn=custom_collate, batch_size=conf['batch_size'], shuffle=True)
        val_dataset = ComplexEventDataset(conf, val_data, val_graph_dict)
        self.val_loader = DataLoader(
            val_dataset, num_workers=4, collate_fn=custom_collate, batch_size=conf['batch_size'], shuffle=False)
        test_dataset = ComplexEventDataset(conf, test_data, test_graph_dict)
        self.test_loader = DataLoader(
            test_dataset, num_workers=4, collate_fn=custom_collate, batch_size=conf['batch_size'], shuffle=False)

    def load_data(self, path):
        with open(path + '/train.txt', 'r') as f:
            train_data = f.readlines()
        with open(path + '/train_graphs.pkl', 'rb') as f:
            train_graph_dict = pickle.load(f)
        with open(path + '/val.txt', 'r') as f:
            val_data = f.readlines()
        with open(path + '/val_graphs.pkl', 'rb') as f:
            val_graph_dict = pickle.load(f)
        with open(path + '/test.txt', 'r') as f:
            test_data = f.readlines()
        with open(path + '/test_graphs.pkl', 'rb') as f:
            test_graph_dict = pickle.load(f)
        return train_data, train_graph_dict, val_data, val_graph_dict, test_data, test_graph_dict
