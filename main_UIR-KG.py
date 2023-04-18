import os
import sys
import random
from time import time
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from UIR_KG import UIR_KG
from Parser import *
from utils.log_helper import *
from data_loader import DataLoader


def evaluate(model, data, Ks, device):
    test_batch_size = data.test_batch_size
    train_user_dict = data.train_user_dict
    test_user_dict = data.test_user_dict
    train_item_dict = data.train_item_dict

    model.eval()  # 模型变为评估模式，torch.nn.Dropout等模块会失效

    user_ids = list(test_user_dict.keys())
    # 将训练集中所有用户id按test_batch_size进行分组，每一组都是长度为test_batch_size 元素为user_id的list
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = data.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg', 'AD', 'MD', 'ARP']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    AD_dict = {k: set() for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')

            batch_scores = batch_scores.cpu()
            batch_metrics = cal_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
                                             batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, train_item_dict)
            cf_scores.append(batch_scores.numpy())

            for k in Ks:
                for m in metric_names:
                    if m == 'AD':
                        AD_dict[k] = set.union(AD_dict[k],batch_metrics[k][m])
                    else:
                        metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)

    for k in Ks:
        for m in metric_names:
            if m == 'AD':
                metrics_dict[k][m] = len(AD_dict[k])
            elif m == 'MD' or m == 'ARP':
                metrics_dict[k][m] = np.mean(metrics_dict[k][m])
            else:
                metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()

    return cf_scores, metrics_dict

# def evaluate(model, dataloader, Ks, device):
#     test_batch_size = dataloader.test_batch_size
#     train_user_dict = dataloader.train_user_dict
#     test_user_dict = dataloader.test_user_dict
#     train_item_dict = dataloader.train_item_dict
#
#     model.eval()
#
#     user_ids = list(test_user_dict.keys())
#     # 按照 每个batch长度为test_batch_size，将训练集中的用户id分为多个batch
#     user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
#     user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
#
#     n_items = dataloader.n_items
#     item_ids = torch.arange(n_items, dtype=torch.long).to(device)
#
#     cf_scores = []
#     metric_names = ['precision', 'recall', 'ndcg', 'AD', 'MD', 'ARP']  #
#     metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
#
#     ad_dict = {k: set() for k in Ks}
#
#     with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
#         for batch_user_ids in user_ids_batches:
#             batch_user_ids = batch_user_ids.to(device)
#
#             with torch.no_grad():
#                 batch_scores = model(batch_user_ids, item_ids, mode='predict')  # (n_batch_users, n_items)
#
#             batch_scores = batch_scores.cpu()
#             batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, train_item_dict,
#                                              batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
#
#             cf_scores.append(batch_scores.numpy())
#             for k in Ks:
#                 for m in metric_names:
#                     if m == 'AD':
#                         ad_dict[k] = set.union(ad_dict[k], batch_metrics[k][m])
#                     else:
#                         metrics_dict[k][m].append(batch_metrics[k][m])
#             pbar.update(1)
#
#     cf_scores = np.concatenate(cf_scores, axis=0)
#     for k in Ks:
#         for m in metric_names:
#             if m == 'AD':
#                 metrics_dict[k][m] = len(ad_dict[k])
#             elif m == 'MD' or m == 'ARP':
#                 metrics_dict[k][m] = np.mean(metrics_dict[k][m])
#             else:
#                 metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
#
#     return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)  # args.seed = 2019
    np.random.seed(args.seed)  # args.seed = 2019
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # load data
    data = DataLoader(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer                                                                                ,data.train_user_dict3 在A_in之前
    model = UIR_KG(args, data.n_users, data.n_entities, data.n_relations, data.train_user_dict, data.train_user_dict2,
                   data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)  # args.lr： default=0.0001
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)  # [20, 40, 60, 80, 100]
    k_min = min(Ks)
    k_40 = Ks[1]
    k_60 = Ks[2]
    k_80 = Ks[3]
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': [], 'AD': [], 'MD': [], 'ARP': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        #             3507140              1024
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict,
                                                                                         data.train_user_dict2,
                                                                                         data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                          n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()

            # if (iter + 1) % 100 == 0 or iter == n_cf_batch:
            cf_optimizer.step()
            cf_optimizer.zero_grad()

            cf_total_loss += cf_batch_loss.item()

            torch.cuda.empty_cache()

            if (iter % args.cf_print_every) == 0:
                logging.info(
                    'CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info(
            'CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                              n_cf_batch,
                                                                                                              time() - time1,
                                                                                                              cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_ckg_train_data // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                data.train_ckg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail,
                                  mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                          n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()

            # if (iter +1) % 100 == 0 or iter == n_kg_batch:
            kg_optimizer.step()
            kg_optimizer.zero_grad()

            kg_total_loss += kg_batch_loss.item()

            torch.cuda.empty_cache()

            if (iter % args.kg_print_every) == 0:
                logging.info(
                    'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info(
            'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                              n_kg_batch,
                                                                                                              time() - time3,
                                                                                                              kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)

            logging.info(
                'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], AD [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}],MD [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}], ARP [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(
                    epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_40]['precision'],
                    metrics_dict[k_60]['precision'], metrics_dict[k_80]['precision'], metrics_dict[k_max]['precision'],
                    metrics_dict[k_min]['recall'], metrics_dict[k_40]['recall'], metrics_dict[k_60]['recall'],
                    metrics_dict[k_80]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'],
                    metrics_dict[k_40]['ndcg'], metrics_dict[k_60]['ndcg'], metrics_dict[k_80]['ndcg'],
                    metrics_dict[k_max]['ndcg'], metrics_dict[k_min]['AD'], metrics_dict[k_40]['AD'],
                    metrics_dict[k_60]['AD'], metrics_dict[k_80]['AD'], metrics_dict[k_max]['AD'],
                    metrics_dict[k_min]['MD'], metrics_dict[k_40]['MD'], metrics_dict[k_60]['MD'],
                    metrics_dict[k_80]['MD'], metrics_dict[k_max]['MD'], metrics_dict[k_min]['ARP'],
                    metrics_dict[k_40]['ARP'], metrics_dict[k_60]['ARP'], metrics_dict[k_80]['ARP'],
                    metrics_dict[k_max]['ARP']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg', 'AD', 'MD', 'ARP']:  #
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

        torch.cuda.empty_cache()

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg', 'AD', 'MD', 'ARP']:  #
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info(
        'Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}], AD [{:.4f}, {:.4f}], MD [{:.4f}, {:.4f}], ARP [{:.4f}, {:.4f}]'.format(
            #
            int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)],
            best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)],
            best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)],
            best_metrics['ndcg@{}'.format(k_max)], best_metrics['AD@{}'.format(k_min)],
            best_metrics['AD@{}'.format(k_max)], best_metrics['MD@{}'.format(k_min)],
            best_metrics['MD@{}'.format(k_max)], best_metrics['ARP@{}'.format(k_min)],
            best_metrics['ARP@{}'.format(k_max)]))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def cal_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks, item_train_dict):
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)

    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        # torch.sort() 返回值有两个，第一个为排好序的tensor，第二个是原tensor元素的index
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)

    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)
    rank_indices = rank_indices.numpy()
    # binary_hit.shape is (len(user_ids),n_items)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]["precision"] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]["recall"] = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]["ndcg"] = ndcg_at_k_batch(binary_hit, k)
        metrics_dict[k]["AD"] = AD_at_k_batch(rank_indices, k)  # type is set
        metrics_dict[k]["MD"] = MD_at_k_batch(rank_indices, k)
        metrics_dict[k]["ARP"] = ARP_at_k_batch(rank_indices, k, item_train_dict)
    return metrics_dict

# def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, train_item_dict, user_ids, item_ids, Ks):
#     """
#     cf_scores: # (n_batch_users, n_items)
#     """
#     ## user_ids是长度为test_batch_size的用户id表，item_ids是全部物品
#     test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)#创建一个矩阵，元素全为0
#     for idx, u in enumerate(user_ids):
#         train_pos_item_list = train_user_dict[u]
#         test_pos_item_list = test_user_dict[u]
#         cf_scores[idx][train_pos_item_list] = -np.inf
#         test_pos_item_binary[idx][test_pos_item_list] = 1#在测试集中有过的交互，其在矩阵中对应位置的元素赋值为1
#
#     try:#cf_scores 每一行从大到小排列，rank_indices为原tensor中元素对应索引值
#         _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
#     except:
#         _, rank_indices = torch.sort(cf_scores, descending=True)
#
#     rank_indices = rank_indices.cpu() # (n_batch_users, n_items)
#
#
#     binary_hit = []
#     for i in range(len(user_ids)):
#         binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
#     binary_hit = np.array(binary_hit, dtype=np.float32)
#     rank_indices = rank_indices.numpy()
#
#     metrics_dict = {}
#     for k in Ks:
#         metrics_dict[k] = {}
#         metrics_dict[k]["precision"] = precision_at_k_batch(binary_hit, k)
#         metrics_dict[k]["recall"] = recall_at_k_batch(binary_hit, k)
#         metrics_dict[k]["ndcg"] = ndcg_at_k_batch(binary_hit, k)
#         metrics_dict[k]["AD"] = AD_at_k_batch(rank_indices, k)  # type is set
#         metrics_dict[k]["MD"] = MD_at_k_batch(rank_indices, k)
#         metrics_dict[k]["ARP"] = ARP_at_k_batch(rank_indices, k, train_item_dict)
#     return metrics_dict


def precision_at_k_batch(binary_hit, k):
    """
        calculate Precision@k
        hits: array, element is binary (0 / 1), 2-dim
        """
    res = binary_hit[:, :k].mean(axis=1)
    return res


def recall_at_k_batch(binary_hit, k):
    """
        calculate Recall@k
        hits: array, element is binary (0 / 1), 2-dim
        """
    res = (binary_hit[:, :k].sum(axis=1) / binary_hit.sum(axis=1))
    return res


def ndcg_at_k_batch(binary_hit, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = binary_hit[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(binary_hit), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def AD_at_k_batch(rank_indices, k):
    all_batch_rec = rank_indices[:, :k].flatten().tolist()
    all_batch_rec = set(all_batch_rec)
    return all_batch_rec


def MD_at_k_batch(rank_indices, k):
    n_users = rank_indices.shape[0]
    n_Co_occur = 0
    for i in range(n_users - 1):
        for j in range(i + 1, n_users):
            merge_array = np.vstack((rank_indices[i][:k], rank_indices[j][:k])).flatten()
            de_duplication_merge_array = np.unique(merge_array)
            n_Co_occur += len(merge_array) - len(de_duplication_merge_array)
    return 1 - (2 * n_Co_occur) / (k * n_users * (n_users - 1))


def ARP_at_k_batch(rank_indices, k, item_dict):
    all_batch_rec = rank_indices[:, :k].flatten().tolist()
    num_populary = 0
    for item_id in all_batch_rec:
        num_populary += len(item_dict[item_id])
    return num_populary / len(all_batch_rec)


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


if __name__ == '__main__':
    args = parse_args()
    train(args)
    # predict(args)
