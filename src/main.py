import argparse
import json
import yaml
import numpy as np
import os
from re_net import RENet
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataloader import Datasets

from torch.utils.tensorboard import SummaryWriter


def main(args):

    # check cuda and set random seeds
    device = torch.device("cuda:{}".format(
        conf['gpu']) if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    # seed = 999
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # load data
    dataset = Datasets(conf)
    conf['num_nodes'] = dataset.num_nodes
    conf['num_rels'] = dataset.num_rels

    log_path = "./logs/{}".format(conf["dataset"])
    run_path = "./runs/{}".format(conf["dataset"])
    checkpoint_model_path = "./checkpoints/{}/model".format(conf["dataset"])
    checkpoint_conf_path = "./checkpoints/{}/conf".format(conf["dataset"])

    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
    if not os.path.isdir(checkpoint_conf_path):
        os.makedirs(checkpoint_conf_path)

    settings = []
    if conf["info"] != "":
        settings += [conf["info"]]
    settings += [str(conf['RGCN_bases']), str(conf['n_hidden'])]
    settings += ["lr" + str(conf['lr']), "wd" +
                 str(conf['wd']), "clambda" + str(conf['c_lambda'])]
    setting = "_".join(settings)

    log_path = log_path + "/" + setting
    run_path = run_path + "/" + setting
    checkpoint_model_path = checkpoint_model_path + "/" + setting
    #checkpoint_conf_path = checkpoint_conf_path + "/" + setting

    run = SummaryWriter(run_path)

    print("start training...")
    model = RENet(conf)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=conf['lr'], weight_decay=conf['wd'])

    log = open(log_path, "a")
    log.write(str(conf) + "\n")
    log.close()

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])
    best_epoch = 0
    best_mrr = 0
    for epoch in range(conf["max_epochs"]):

        epoch_anchor = epoch * batch_cnt

        pbar = tqdm(enumerate(dataset.train_loader),
                    total=len(dataset.train_loader))
        for batch_idx, batch in pbar:

            model.train()

            batch_anchor = epoch_anchor + batch_idx

            optimizer.zero_grad()

            batch_data = batch['s']
            batch_data = [x.to(device) for x in batch_data]
            ce_loss_s, c_loss_s = model(batch_data, subject=True)

            batch_data = batch['o']
            batch_data = [x.to(device) for x in batch_data]
            ce_loss_o, c_loss_o = model(batch_data, subject=False)

            ce_loss = (ce_loss_s + ce_loss_o) / 2
            c_loss = (c_loss_s + c_loss_o) / 2
            loss = ce_loss + conf["c_lambda"] * c_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), conf['grad_norm'])  # clip gradients
            optimizer.step()

            run.add_scalar('loss/loss_all', loss.item(), batch_anchor)
            run.add_scalar('loss/ce_loss', ce_loss.item(), batch_anchor)
            run.add_scalar('loss/ce_loss: sub->obj',
                           ce_loss_s.item(), batch_anchor)
            run.add_scalar('loss/ce_loss: obj->sub',
                           ce_loss_o.item(), batch_anchor)
            if conf['use_contrastive']:
                run.add_scalar('loss/c_loss', c_loss.item(), batch_anchor)
                run.add_scalar('loss/c_loss: sub->obj',
                               c_loss_s.item(), batch_anchor)
                run.add_scalar('loss/c_loss: obj->sub',
                               c_loss_o.item(), batch_anchor)

            pbar.set_description("epoch: %d, loss: %.4f" %
                                 (epoch, loss.item()))

            if (batch_anchor+1) % test_interval_bs == 0:
                metrics = {}

                print("start evaluating...")
                metrics["val"] = test(model, dataset.val_loader, conf)

                print("\n")

                print("start testing...")
                metrics["test"] = test(model, dataset.test_loader, conf)

                if metrics["val"][0] > best_mrr:
                    best_mrr = metrics["val"][0]
                    torch.save(model.state_dict(), checkpoint_model_path)
                    best_epoch = epoch
                    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    best_perform_val = "{}, Val: Best in epoch {}, MRR={:.4f}, Hits@1={:.4f}, Hits@3={:.4f}, Hits@10={:.4f}".format(cur_time,
                                                                                                                                    best_epoch,
                                                                                                                                    metrics[
                                                                                                                                        "val"][0],
                                                                                                                                    metrics[
                                                                                                                                        "val"][1],
                                                                                                                                    metrics[
                                                                                                                                        "val"][2],
                                                                                                                                    metrics["val"][3])

                    best_perform_test = "{}, Test: Best in epoch {}, MRR={:.4f}, Hits@1={:.4f}, Hits@3={:.4f}, Hits@10={:.4f}".format(cur_time,
                                                                                                                                      best_epoch,
                                                                                                                                      metrics[
                                                                                                                                          "test"][0],
                                                                                                                                      metrics[
                                                                                                                                          "test"][1],
                                                                                                                                      metrics[
                                                                                                                                          "test"][2],
                                                                                                                                      metrics["test"][3])

                    log = open(log_path, "a")

                    print(best_perform_val)
                    print(best_perform_test)
                    log.write(best_perform_val + "\n")
                    log.write(best_perform_test + "\n")

                    log.close()

                write_log(run, log_path, batch_anchor, metrics)

    print("training done")


def test(model, dataloader, conf):

    device = conf['device']
    model.eval()
    # Compute raw/filter Mean Reciprocal Rank (MRR) and Hits@1/3/10.
    result = torch.tensor([0, 0, 0, 0], dtype=torch.float)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in pbar:

        with torch.no_grad():

            batch_data = batch['s']
            batch_data = batch_data = [
                x.to(device) if x is not None else x for x in batch_data]
            obj_pred, groundtruth_obj = model(
                batch_data, subject=True, return_prob=True)

            batch_data = batch['o']
            batch_data = batch_data = [
                x.to(device) if x is not None else x for x in batch_data]
            sub_pred, groundtruth_sub = model(
                batch_data, subject=False, return_prob=True)

        result += model.test(obj_pred, groundtruth_obj) * \
            groundtruth_obj.size(0)
        result += model.test(sub_pred, groundtruth_sub) * \
            groundtruth_sub.size(0)

    result = result / (2 * len(dataloader.dataset))

    return result.tolist()


def write_log(run, log_path, step, metrics):
    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_res = metrics["val"]
    test_res = metrics["test"]
    res2metric = {0: 'MRR', 1: 'hits@1', 2: 'hits@3', 3: 'hits@10'}

    for index, score in enumerate(val_res):
        run.add_scalar("Val/{}".format(res2metric[index]), score, step)
    for index, score in enumerate(test_res):
        run.add_scalar("test/{}".format(res2metric[index]), score, step)

    val_str = "{}, Val: MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}".format(cur_time,
                                                                                             val_res[0],
                                                                                             val_res[1],
                                                                                             val_res[2],
                                                                                             val_res[3])
    test_str = "{}, Test: MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}".format(cur_time,
                                                                                               test_res[0],
                                                                                               test_res[1],
                                                                                               test_res[2],
                                                                                               test_res[3])

    log = open(log_path, "a")
    log.write("{}\n".format(val_str))
    log.write("{}\n".format(test_str))
    log.close()

    print(val_str)
    print(test_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--dataset", type=str, default="GDELT_EG",
                        help="dataset")
    parser.add_argument("--info", type=str, default="",
                        help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    dataset_name = args.dataset
    conf = yaml.safe_load(open("./config.yaml"))
    conf = conf[dataset_name]

    conf['gpu'] = args.gpu
    conf['dataset'] = dataset_name
    conf['info'] = args.info
    print(conf)
    main(conf)
