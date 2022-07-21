import numpy as np
import os
import pickle
import dgl
import torch
from tqdm import tqdm
import argparse


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


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


def get_data_with_t_c(data, cid, tim):
    x = data[np.where((data[:, 3] == tim) & (data[:, 4] == cid))].copy()
    x = np.delete(x, [3, 4], 1)  # drops 3rd column
    return x


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g

def main(args):

    graph_dict_train = {}
    graph_dict_dev = {}
    graph_dict_test = {}

    train_data, train_times, train_cids = load_quadruples(args.path, 'train.txt')
    test_data, test_times, test_cids = load_quadruples(args.path, 'test.txt')
    dev_data, dev_times, dev_cids = load_quadruples(args.path, 'val.txt')
    # total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

    num_e, num_r = get_total_number(args.path, 'stat.txt')

    s_his = [[] for _ in range(num_e)]
    o_his = [[] for _ in range(num_e)]
    s_his_t = [[] for _ in range(num_e)]
    o_his_t = [[] for _ in range(num_e)]
    s_history_data = [[] for _ in range(len(train_data))]
    o_history_data = [[] for _ in range(len(train_data))]
    s_history_data_t = [[] for _ in range(len(train_data))]
    o_history_data_t = [[] for _ in range(len(train_data))]
    e = []
    r = []
    latest_t,latest_c = 0,0
    s_his_cache = [[] for _ in range(num_e)]
    o_his_cache = [[] for _ in range(num_e)]
    s_his_cache_t = [None for _ in range(num_e)]
    o_his_cache_t = [None for _ in range(num_e)]

    print("Generating graphs for training..")
    with tqdm(total=len(train_cids) * len(train_times)) as pbar:
        for cid in train_cids:
            graph_dict_train[cid] = {}
            for tim in train_times:
                # print(str(tim) + '\t' + str(max(train_times)))
                data = get_data_with_t_c(train_data, cid, tim)
                graph_dict_train[cid][tim] = (get_big_graph(data, num_r))
                pbar.update(1)

    print("Generating graphs for evaluation..")
    with tqdm(total=len(dev_cids) * len(dev_times)) as pbar:
        for cid in dev_cids:
            graph_dict_dev[cid] = {}
            for tim in dev_times:
                # print(str(tim) + '\t' + str(max(train_times)))
                data = get_data_with_t_c(dev_data, cid, tim)
                graph_dict_dev[cid][tim] = (get_big_graph(data, num_r))
                pbar.update(1)

    print("Generating graphs for testing..")
    with tqdm(total=len(test_cids) * len(test_times)) as pbar:
        for cid in test_cids:
            graph_dict_test[cid] = {}
            for tim in test_times:
                # print(str(tim) + '\t' + str(max(train_times)))
                data = get_data_with_t_c(test_data, cid, tim)
                graph_dict_test[cid][tim] = (get_big_graph(data, num_r))
                pbar.update(1)

    with open(os.path.join(args.path, 'train_graphs.txt'), 'wb') as fp:
        pickle.dump(graph_dict_train, fp)

    with open(os.path.join(args.path, 'val_graphs.txt'), 'wb') as fp:
        pickle.dump(graph_dict_dev, fp)

    with open(os.path.join(args.path, 'test_graphs.txt'), 'wb') as fp:
        pickle.dump(graph_dict_test, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graphs')
    parser.add_argument("--path", type=str, default="data",
                        help="input data path")  
    args = parser.parse_args()

    main(args)

# history_len = 14
# print("Gathering history for training set..")
# for i, train in tqdm(enumerate(train_data)):
#     '''if i % 10000 == 0:
#         print("train", i, len(train_data))'''
#     # if i == 10000:
#     #     break
#     cid = train[4]
#     if latest_c != cid:
#         s_his = [[] for _ in range(num_e)]
#         o_his = [[] for _ in range(num_e)]
#         s_his_t = [[] for _ in range(num_e)]
#         o_his_t = [[] for _ in range(num_e)]
#         s_his_cache = [[] for _ in range(num_e)]
#         o_his_cache = [[] for _ in range(num_e)]
#         s_his_cache_t = [None for _ in range(num_e)]
#         o_his_cache_t = [None for _ in range(num_e)]
#         latest_c = cid

#     t = train[3]
#     if latest_t != t:
#         for ee in range(num_e):
#             if len(s_his_cache[ee]) != 0:
#                 if len(s_his[ee]) >= history_len:
#                     s_his[ee].pop(0)
#                     s_his_t[ee].pop(0)

#                 s_his[ee].append(s_his_cache[ee].copy())
#                 s_his_t[ee].append(s_his_cache_t[ee])
#                 s_his_cache[ee] = []
#                 s_his_cache_t[ee] = None
#             if len(o_his_cache[ee]) != 0:
#                 if len(o_his[ee]) >= history_len:
#                     o_his[ee].pop(0)
#                     o_his_t[ee].pop(0)

#                 o_his[ee].append(o_his_cache[ee].copy())
#                 o_his_t[ee].append(o_his_cache_t[ee])
#                 o_his_cache[ee] = []
#                 o_his_cache_t[ee] = None
#         latest_t = t
#     s = train[0]
#     r = train[1]
#     o = train[2]
#     cid = train[4]
#     # print(s_his[r][s])
#     s_history_data[i] = s_his[s].copy()
#     o_history_data[i] = o_his[o].copy()
#     s_history_data_t[i] = s_his_t[s].copy()
#     o_history_data_t[i] = o_his_t[o].copy()
#     # print(o_history_data_g[i])

#     if len(s_his_cache[s]) == 0:
#         s_his_cache[s] = np.array([[r, o]])
#     else:
#         s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
#     s_his_cache_t[s] = (cid, t)

#     if len(o_his_cache[o]) == 0:
#         o_his_cache[o] = np.array([[r, s]])
#     else:
#         o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
#     o_his_cache_t[o] = (cid, t)

#     # print(s_history_data[i], s_history_data_g[i])
#     # with open('ttt.txt', 'wb') as fp:
#     #     pickle.dump(s_history_data_g, fp)
#     # print("save")



#     # graph.keys() -> time
#     # modified_graph.keys_1() -> complex_id
#     # modified_graph.keys_2() -> time
#     # done

# with open('data/train_history_sub.txt', 'wb') as fp:
#     # s_history_data[0].__len__() = self.seq_len -> array[[h,h,h,h], [h,h,h,h], ...]
#     # s_history_data_t[0].__len__() = self.seq_len -> array[(c, t1), (c, t2), (c, t2), ..]
#     # sub [1024] h_sub [<1024*self.seq_len, embedding_dim]

#     # dataloader -> __getitem()__: [sub, rel, obj, c, t]
#     # dataloader -> __getitem()__: [sub, rel, obj, c, t, history]

#     #pickle.dump([s_history_data, s_history_data_t], fp)
#     pickle.dump(s_history_data_t, fp)
# with open('data/train_history_ob.txt', 'wb') as fp:
#     pickle.dump(o_history_data_t, fp)

# # print(s_history_data[0])
# s_history_data_dev = [[] for _ in range(len(dev_data))]
# o_history_data_dev = [[] for _ in range(len(dev_data))]
# s_history_data_dev_t = [[] for _ in range(len(dev_data))]
# o_history_data_dev_t = [[] for _ in range(len(dev_data))]
# print("Gathering history for evaluation set..")
# for i, dev in tqdm(enumerate(dev_data)):
#     '''if i % 10000 == 0:
#         print("valid", i, len(dev_data))'''
#     cid = dev[4]
#     if latest_c != cid:
#         s_his = [[] for _ in range(num_e)]
#         o_his = [[] for _ in range(num_e)]
#         s_his_t = [[] for _ in range(num_e)]
#         o_his_t = [[] for _ in range(num_e)]
#         s_his_cache = [[] for _ in range(num_e)]
#         o_his_cache = [[] for _ in range(num_e)]
#         s_his_cache_t = [None for _ in range(num_e)]
#         o_his_cache_t = [None for _ in range(num_e)]
#         latest_c = cid
#     t = dev[3]
#     if latest_t != t:
#         for ee in range(num_e):
#             if len(s_his_cache[ee]) != 0:
#                 if len(s_his[ee]) >= history_len:
#                     s_his[ee].pop(0)
#                     s_his_t[ee].pop(0)
#                 s_his_t[ee].append(s_his_cache_t[ee])
#                 s_his[ee].append(s_his_cache[ee].copy())
#                 s_his_cache[ee] = []
#                 s_his_cache_t[ee] = None
#             if len(o_his_cache[ee]) != 0:
#                 if len(o_his[ee]) >= history_len:
#                     o_his[ee].pop(0)
#                     o_his_t[ee].pop(0)

#                 o_his_t[ee].append(o_his_cache_t[ee])
#                 o_his[ee].append(o_his_cache[ee].copy())

#                 o_his_cache[ee] = []
#                 o_his_cache_t[ee] = None
#         latest_t = t
#     s = dev[0]
#     r = dev[1]
#     o = dev[2]
#     s_history_data_dev[i] = s_his[s].copy()
#     o_history_data_dev[i] = o_his[o].copy()
#     s_history_data_dev_t[i] = s_his_t[s].copy()
#     o_history_data_dev_t[i] = o_his_t[o].copy()
#     if len(s_his_cache[s]) == 0:
#         s_his_cache[s] = np.array([[r, o]])
#     else:
#         s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
#     s_his_cache_t[s] = (cid, t)

#     if len(o_his_cache[o]) == 0:
#         o_his_cache[o] = np.array([[r, s]])
#     else:
#         o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
#     o_his_cache_t[o] = (cid, t)

#     # print(o_his_cache[o])

# with open('data/dev_history_sub.txt', 'wb') as fp:
#     #pickle.dump([s_history_data_dev, s_history_data_dev_t], fp)
#     pickle.dump(s_history_data_dev_t, fp)
# with open('data/dev_history_ob.txt', 'wb') as fp:
#     #pickle.dump([o_history_data_dev, o_history_data_dev_t], fp)
#     pickle.dump(o_history_data_dev_t, fp)

# s_history_data_test = [[] for _ in range(len(test_data))]
# o_history_data_test = [[] for _ in range(len(test_data))]

# s_history_data_test_t = [[] for _ in range(len(test_data))]
# o_history_data_test_t = [[] for _ in range(len(test_data))]
# print("Gathering history for testing set..")
# for i, test in tqdm(enumerate(test_data)):
#     '''if i % 10000 == 0:
#         print("test", i, len(test_data))'''
#     cid = test[4]
#     if latest_c != cid:
#         s_his = [[] for _ in range(num_e)]
#         o_his = [[] for _ in range(num_e)]
#         s_his_t = [[] for _ in range(num_e)]
#         o_his_t = [[] for _ in range(num_e)]
#         s_his_cache = [[] for _ in range(num_e)]
#         o_his_cache = [[] for _ in range(num_e)]
#         s_his_cache_t = [None for _ in range(num_e)]
#         o_his_cache_t = [None for _ in range(num_e)]
#         latest_c = cid
#     t = test[3]
#     if latest_t != t:
#         for ee in range(num_e):
#             if len(s_his_cache[ee]) != 0:
#                 if len(s_his[ee]) >= history_len:
#                     s_his[ee].pop(0)
#                     s_his_t[ee].pop(0)
#                 s_his_t[ee].append(s_his_cache_t[ee])

#                 s_his[ee].append(s_his_cache[ee].copy())
#                 s_his_cache[ee] = []
#                 s_his_cache_t[ee] = None
#             if len(o_his_cache[ee]) != 0:
#                 if len(o_his[ee]) >= history_len:
#                     o_his[ee].pop(0)
#                     o_his_t[ee].pop(0)

#                 o_his_t[ee].append(o_his_cache_t[ee])

#                 o_his[ee].append(o_his_cache[ee].copy())
#                 o_his_cache[ee] = []
#                 o_his_cache_t[ee] = None
#         latest_t = t
#     s = test[0]
#     r = test[1]
#     o = test[2]

#     s_history_data_test[i] = s_his[s].copy()
#     o_history_data_test[i] = o_his[o].copy()
#     s_history_data_test_t[i] = s_his_t[s].copy()
#     o_history_data_test_t[i] = o_his_t[o].copy()
#     if len(s_his_cache[s]) == 0:
#         s_his_cache[s] = np.array([[r, o]])
#     else :
#         s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
#     s_his_cache_t[s] = (cid, t)

#     if len(o_his_cache[o]) == 0:
#         o_his_cache[o] = np.array([[r, s]])
#     else :
#         o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
#     o_his_cache_t[o] = (cid, t)
#     # print(o_his_cache[o])

# with open('data/test_history_sub.txt', 'wb') as fp:
#     pickle.dump(s_history_data_test_t, fp)
#     #pickle.dump([s_history_data_test, s_history_data_test_t], fp)
# with open('data/test_history_ob.txt', 'wb') as fp:
#     #pickle.dump([o_history_data_test, o_history_data_test_t], fp)
#     pickle.dump(o_history_data_test_t, fp)
#     # print(train)
