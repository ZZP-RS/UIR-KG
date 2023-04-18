import collections
import random

import torch
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DataLoader(object):

    def __init__(self, args, logging):
        self.args = args
        self.dataset = args.dataset
        self.use_pretrain = args.use_pretrain
        self.pretrain_embeddings_dir = args.pretrain_embeddings_dir

        # datasets\last-fm
        self.data_dir = os.path.join(args.dataset_dir, args.dataset)
        self.train_dir = os.path.join(self.data_dir, 'Neighbor_selector.txt')
        self.train_dir2 = os.path.join(self.data_dir, 'train.txt')
        self.test_dir = os.path.join(self.data_dir, 'test.txt')
        self.kg_dir = os.path.join(self.data_dir, 'kg_final.txt')

        # load data. #train_data type is tulpe, len is 2
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_dir)

        self.cf_train_data2, self.train_user_dict2 = self.load_cf(self.train_dir2)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_dir)

        self.train_item_dict = self._get_item_dict(self.cf_train_data2)
        self.param_cf()

        # load pretrain embeddings
        if self.use_pretrain == 1:
            self.load_pretrained_data()

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        # type : pandas.core.frame.DataFrame  464567 rows x 3 columns 464567
        kg_data = self.load_kg(self.kg_dir)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def _get_item_dict(self, train_data):
        item_dict = collections.defaultdict(list)
        assert len(train_data[0]) == len(train_data[1])
        for i in range(len(train_data[0])):
            if train_data[1][i] not in item_dict.keys():
                item_dict[train_data[1][i]] = [train_data[0][i]]
            else:
                item_dict[train_data[1][i]].append(train_data[0][i])
        return item_dict

    def load_cf(self, filename):
        userID_list = []
        itemID_list = []
        user_dict = dict()

        lines = open(filename, "r").readlines()
        for l in lines:
            temp = l.strip()
            interaction = [int(i) for i in temp.split()]

            if len(interaction) > 1:
                user_id, item_ids = interaction[0], interaction[1:]
                # deduplicate
                item_ids = list(set(item_ids))
                for item_id in item_ids:
                    userID_list.append(user_id)
                    itemID_list.append(item_id)
                user_dict[user_id] = item_ids
        userID_array = np.array(userID_list, dtype=np.int32)
        itemID_array = np.array(itemID_list, dtype=np.int32)
        return (userID_array, itemID_array), user_dict

    def param_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1  # 23566
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1  # 48123
        self.n_cf_train = len(self.cf_train_data[0])  # 1289003
        self.n_cf_test = len(self.cf_test_data[0])  # 423635

    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embeddings_dir, self.dataset, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embedding_dim
        assert self.item_pre_embed.shape[1] == self.args.embedding_dim

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_data(self, kg_data):
        # inverse kg_data, double the number of relation
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # insert 'interation' relation
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # user_id shift self.n_entities
        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
                             self.cf_train_data[1].astype(np.int32))

        # user id 向右偏移self.n_entities
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                self.train_user_dict.items()}
        self.train_user_dict2 = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                 self.train_user_dict2.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                               self.test_user_dict.items()}

        # add interactions to kg_data
        cf_train_data_to_triplet = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf_train_data_to_triplet['h'] = self.cf_train_data[0]
        cf_train_data_to_triplet['t'] = self.cf_train_data[1]

        # add inverse interaction to kg_data
        inverse_cf_train_data_to_triplet = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32),
                                                        columns=['h', 'r', 't'])
        inverse_cf_train_data_to_triplet['h'] = self.cf_train_data[1]
        inverse_cf_train_data_to_triplet['t'] = self.cf_train_data[0]

        # concat kg_data and interaction data
        self.ckg_train_data = pd.concat([kg_data, cf_train_data_to_triplet, inverse_cf_train_data_to_triplet],
                                        ignore_index=True)
        self.n_ckg_train_data = len(self.ckg_train_data)

        h_list = []
        t_list = []
        r_list = []

        # construct dict, key is h,value is a list of (t,r)
        self.train_ckg_dict = collections.defaultdict(list)
        # key is r, value is a list of (h,t)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.ckg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_ckg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)
        logging.info('n_kg_train:        %d' % self.n_ckg_train_data)

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [h_t[0] for h_t in ht_list]
            cols = [h_t[1] for h_t in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            def symmetric_norm_lap(adj):
                rowsum = np.array(adj.sum(axis=1))  # 行求和 变成列向量，元素为节点的度

                d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 将rowsum中的每个元素求它的-0.5次方，然后平铺成一维数组
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # 将所有的无穷值赋0
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 生成对角矩阵

                norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)  # 归一化
                return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))  # 行求和 变成列向量，元素为节点的度

            d_inv = np.power(rowsum, -1.0).flatten()  # 将rowsum中的每个元素求它的-1次方，然后平铺成一维数组
            d_inv[np.isinf(d_inv)] = 0  # 将所有的无穷值赋0
            d_mat_inv = sp.diags(d_inv)  # 生成对角矩阵

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())  # 将len(self.laplacian_dict.keys())个拉普拉斯矩阵对应元素相加
        self.A_in = self.convert_coo2tensor(A_in.tocoo())  # 将A_in.tocoo() 转换为 type为 torch.sparse_coo的稀疏矩阵

    def convert_coo2tensor(self, coo):
        values = coo.data  # 一维数组，元素是稀疏矩阵中非零值的元素，长度n为coo中非零元素个数
        indices = np.vstack((coo.row, coo.col))  # 2×n矩阵，对应values中每一个元素在 coo中的横纵坐标。
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def generate_cf_batch(self, user_dict, user_dict2, batch_size):
        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            # u_all_pos_item.append(torch.LongTensor(user_dict[u].reshape(-1).tolist()))
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, user_dict2, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []

        while True:
            if len(sample_pos_items) >= n_sample_pos_items:
                break

            pos_item_index = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_index]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)

        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_dict2, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]
        pos_items2 = user_dict2[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in pos_items2 and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def sample_pos_t_for_h(self, ckg_dict, h, n_sample):
        t_r = ckg_dict[h]
        n_pos_t_r = len(t_r)

        sample_pos_t = []
        sample_pos_r = []
        while True:
            if len(sample_pos_t) >= n_sample:
                break

            pos_t_r_index = np.random.randint(low=0, high=n_pos_t_r, size=1)[0]
            pos_t_r = t_r[pos_t_r_index]
            if pos_t_r[0] not in sample_pos_t:
                sample_pos_t.append(pos_t_r[0])
                sample_pos_r.append(pos_t_r[1])

        return sample_pos_t, sample_pos_r

    def sample_neg_t_for_h(self, ckg_dict, h, r, n_sample, n_entities_users):

        t_r = ckg_dict[h]
        sample_neg_t = []

        while True:
            if len(sample_neg_t) >= n_sample:
                break
            neg_t = np.random.randint(low=0, high=n_entities_users, size=1)[0]
            if (neg_t, r) not in t_r and neg_t not in sample_neg_t:
                sample_neg_t.append(neg_t)
        return sample_neg_t

    def generate_kg_batch(self, ckg_dict, kg_batch_size, n_entities_users):
        all_h = ckg_dict.keys()
        if kg_batch_size <= len(all_h):
            batch_h = random.sample(all_h, kg_batch_size)
        else:
            batch_h = [random.choice(all_h) for _ in range(kg_batch_size)]

        batch_r, batch_pos_t, batch_neg_t = [], [], []
        for h in batch_h:
            # The number of samples is 1
            pos_t, r = self.sample_pos_t_for_h(ckg_dict, h, 1)
            batch_r += r
            batch_pos_t += pos_t

            batch_neg_t += self.sample_neg_t_for_h(ckg_dict, h, r, 1, n_entities_users)

        batch_h = torch.LongTensor(batch_h)
        batch_r = torch.LongTensor(batch_r)
        batch_pos_t = torch.LongTensor(batch_pos_t)
        batch_neg_t = torch.LongTensor(batch_neg_t)

        return batch_h, batch_r, batch_pos_t, batch_neg_t
