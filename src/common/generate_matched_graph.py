'''
generate graph matching results using model-based embeddings,
codes are optimised for matrix calculation,
modify line 185-190 and `config.yaml` for configuration.
'''

import yaml
import torch
import dgl
import numpy as np
import argparse
import os
from dataloader import GraphDataset
import pickle
from math import sqrt
from tqdm import tqdm
from re_net import RENet

import torch_scatter
import torch_geometric
from torch.utils.data import Dataset, DataLoader


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        cids = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            cid = int(line_split[4])
            quadrupleList.append([head, rel, tail, time, cid])
            times.add(time)

            cids.add(cid)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                cid = int(line_split[4])
                quadrupleList.append([head, rel, tail, time, cid])
                times.add(time)

                cids.add(cid)
    times = list(times)
    times.sort()
    cids = list(cids)
    cids.sort()

    return np.array(quadrupleList), np.asarray(times), np.asarray(cids)


def load_data(path, info):
    event_data, _, _ = load_quadruples(path, info + '.txt')
    with open(path + '/' + info + '_graphs.pkl', 'rb') as f:
        graph_dict = pickle.load(f)
    return event_data, graph_dict


def graph_collate(data):
    graphs, keys = zip(*data)

    complex_graphs = []
    graph_anchors = []

    for graph in graphs:
        # record num of nodes
        graph_anchor = list(map(lambda x: x.number_of_nodes(), graph))
        graph_anchors.append(graph_anchor)
        complex_graphs.append(dgl.batch(graph))

    return complex_graphs, keys, graph_anchors


def graph_pooling(model, complex_graphs, graph_anchors, mode='macro'):

    if mode == 'macro':

        complex_anchor = list(
            map(lambda x: x.number_of_nodes(), complex_graphs))
        embeds_propagated = model.aggregator.get_graph_embed(
            dgl.batch(complex_graphs), model.ent_embeds, reverse=False)
        embeds_split = torch.split(embeds_propagated, complex_anchor)
        graph_embed = list(map(lambda x: x.mean(dim=0), embeds_split))
        graph_embed_tensor = torch.stack(graph_embed, dim=0)

    if mode == 'micro':

        complex_anchor = list(
            map(lambda x: x.number_of_nodes(), complex_graphs))
        embeds_propagated = model.aggregator.get_graph_embed(
            dgl.batch(complex_graphs), model.ent_embeds, reverse=False)
        embeds_split = torch.split(embeds_propagated, complex_anchor)

        graph_embed_tensor = []

        for embed_lookup, complex_graph, graph_anchor in zip(embeds_split, complex_graphs, graph_anchors):
            single_day_embedding = torch.split(embed_lookup, graph_anchor)
            complex_graph_embedding = torch.stack(
                list(map(lambda x: x.mean(dim=0), single_day_embedding)))
            complex_graph_embedding = complex_graph_embedding.mean(dim=0)
            graph_embed_tensor.append(complex_graph_embedding)

        graph_embed_tensor = torch.stack(graph_embed_tensor)

    if mode == 'node_rnn':

        complex_anchor = list(
            map(lambda x: x.number_of_nodes(), complex_graphs))
        embeds_propagated = model.aggregator.get_graph_embed(
            dgl.batch(complex_graphs), model.ent_embeds, reverse=False)
        embeds_split = torch.split(embeds_propagated, complex_anchor)

        graph_embed_tensor = []

        for embed_lookup, complex_graph in zip(embeds_split, complex_graphs):

            node_ids = complex_graph.ndata['id'].squeeze(dim=1)
            node_ids_rerank, node_ids_index = node_ids.sort()
            embed_lookup_rerank = embed_lookup[node_ids_index]
            all_embedding, all_embedding_mask = torch_geometric.utils.to_dense_batch(
                embed_lookup_rerank, node_ids_rerank)

            nodes_count, nodes_index = all_embedding_mask.sum(
                dim=-1).sort(descending=True)
            num_nonzero = len(torch.nonzero(nodes_count))
            nodes_count_nonzero = nodes_count[:num_nonzero]
            all_embedding_rerank = all_embedding[nodes_index][:num_nonzero]

            node_packed_input = torch.nn.utils.rnn.pack_padded_sequence(all_embedding_rerank,
                                                                        nodes_count_nonzero.cpu(),
                                                                        batch_first=True)

            _, node_seq_embed = model.encoder(node_packed_input)
            node_seq_embed = node_seq_embed.squeeze(dim=0).mean(dim=0)
            graph_embed_tensor.append(node_seq_embed)

        graph_embed_tensor = torch.stack(graph_embed_tensor)

    return graph_embed_tensor


def get_all_graph_pooling(model, loader, device, mode='node_rnn'):
    all_graph_pooling = []
    all_key_list = []
    for complex_graphs, keys, graph_anchors in tqdm(loader):
        complex_graphs = [complex_graph.to(device)
                          for complex_graph in complex_graphs]
        all_graph_pooling_single = graph_pooling(
            model, complex_graphs, graph_anchors, mode)
        all_graph_pooling.append(all_graph_pooling_single.detach())
        all_key_list.extend(keys)
    all_graph_pooling = torch.cat(all_graph_pooling, dim=0)

    return all_graph_pooling, all_key_list


def cal_sim_batch(query_graph_pooling, answer_graph_pooling, n_batch, topk=10):
    chunked_tensor = query_graph_pooling.chunk(n_batch)
    query_topk = []
    for query_tensor in chunked_tensor:
        query_sim = torch.nn.functional.cosine_similarity(
            query_tensor[:, :, None], answer_graph_pooling.t()[None, :, :])
        query_topk.append(torch.topk(query_sim, topk)[1])
        del query_sim

    query_topk = torch.cat(query_topk, dim=0)

    return query_topk


def get_sim_mat_batch(train_graph_pooling, val_graph_pooling, test_graph_pooling, n_batch, topk):

    val_match = cal_sim_batch(
        val_graph_pooling, train_graph_pooling, n_batch, topk)
    test_match = cal_sim_batch(
        test_graph_pooling, train_graph_pooling, n_batch, topk)

    train_val_cat = torch.cat((train_graph_pooling, val_graph_pooling), dim=0)
    test_match_cat = cal_sim_batch(
        test_graph_pooling, train_val_cat, n_batch, topk)

    return val_match, test_match, test_match_cat


def load_cplxevnet2id(path):
    cplxevent2id = {}
    with open(os.path.join(path, 'cplxevent2id.txt')) as file:
        cplxevent2id_file = file.readlines()
        for line in cplxevent2id_file:
            line = line.strip().split()
            cplxevent2id[int(line[-1])] = line[0][:8]

    return cplxevent2id


def main(args):
    conf = yaml.safe_load(open("./config.yaml"))
    conf = conf['GDELT_EG']
    conf['num_nodes'], conf['num_rels'] = 50, 237
    device = torch.device("cpu")
    conf['device'] = device
    path = '/mnt/kgir/dense50EG_no_overlap'

    train_data, train_dict = load_data(path, 'train')
    val_data, val_dict = load_data(path, 'val')
    test_data, test_dict = load_data(path, 'test')

    cplxevent2id = load_cplxevnet2id(path)

    train_ds = GraphDataset(train_data, train_dict, cplxevent2id)
    val_ds = GraphDataset(val_data, val_dict, cplxevent2id)
    test_ds = GraphDataset(test_data, test_dict, cplxevent2id)

    train_dl = DataLoader(train_ds, collate_fn=graph_collate,
                          shuffle=False, batch_size=128)
    val_dl = DataLoader(val_ds, collate_fn=graph_collate,
                        shuffle=False, batch_size=128)
    test_dl = DataLoader(test_ds, collate_fn=graph_collate,
                         shuffle=False, batch_size=128)

    device = torch.device('cuda:{}'.format(args.gpu))

    model = RENet(conf)
    model.load_state_dict(torch.load(
        './checkpoints/GDELT_EG/model/64_128_lr0.001_wd1e-05_clambda0.8'))
    model.to(device)
    model.eval()

    train_graph_pooling, train_key_list = get_all_graph_pooling(
        model, train_dl, device, args.mode)
    val_graph_pooling, val_key_list = get_all_graph_pooling(
        model, val_dl, device, args.mode)
    test_graph_pooling, test_key_list = get_all_graph_pooling(
        model, test_dl, device, args.mode)

    train_graph_pooling = train_graph_pooling.to(device)
    val_graph_pooling = val_graph_pooling.to(device)
    test_graph_pooling = test_graph_pooling.to(device)

    val_match, test_match, cat_match = get_sim_mat_batch(train_graph_pooling,
                                                         val_graph_pooling,
                                                         test_graph_pooling,
                                                         n_batch=10,
                                                         topk=args.topk)

    sav_dir = './evaluation/GDELT_EG/{}/'.format(args.mode)
    if not os.path.isdir(sav_dir):
        os.makedirs(sav_dir)

    with open(sav_dir + "train_key_list.pkl", 'wb') as fp:
        pickle.dump(train_key_list, fp)
    with open(sav_dir + "val_key_list.pkl", 'wb') as fp:
        pickle.dump(val_key_list, fp)
    with open(sav_dir + "test_key_list.pkl", 'wb') as fp:
        pickle.dump(test_key_list, fp)

    with open(sav_dir + "val_match_top{}.pkl".format(args.topk), 'wb') as fp:
        pickle.dump(val_match, fp)
    with open(sav_dir + "test_match_top{}.pkl".format(args.topk), 'wb') as fp:
        pickle.dump(test_match, fp)
    with open(sav_dir + "cat_match_top{}.pkl".format(args.topk), 'wb') as fp:
        pickle.dump(cat_match, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='direct evaluate')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--dataset", type=str, default="GDELT_EG",
                        help="dataset")
    parser.add_argument("--info", type=str, default="",
                        help="any auxilary info that will be appended to the log file name")
    parser.add_argument("--mode", type=str, default="node_rnn",
                        help="graph pooling method to use")
    parser.add_argument("--topk", type=int, default=10,
                        help="topk matched graph")
    args = parser.parse_args()

    main(args)
