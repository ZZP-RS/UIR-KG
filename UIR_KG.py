import torch.nn as nn
import torch
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, droupout):
        super(Aggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.droupout = droupout

        self.message_droupout = nn.Dropout(droupout)
        self.activation = nn.LeakyReLU()

        self.selfAttention = nn.MultiheadAttention(self.input_dim, num_heads=1)

        # bi-iteraction Aggregator
        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, ego_embeddings, A_in, train_user_dict):
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        # for key in train_user_dict.keys():
        #     # indices = torch.Tensor(train_user_dict[key]).long().to(
        #     #     torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #     # neighbors_embeddings = torch.index_select(side_embeddings, dim=0, index=indices)
        #     # neighbors_embeddings = torch.cat([side_embeddings[[key], :], neighbors_embeddings], dim=0)
        #     indices = torch.LongTensor(train_user_dict[key]).to(torch.device("cuda"))
        #     neighbors_embeddings = side_embeddings[indices]
        #     user_embeddings = side_embeddings[torch.LongTensor([key]).to(torch.device("cuda"))]
        #     neighbors_embeddings = torch.cat([user_embeddings, neighbors_embeddings], dim=0)
        #
        #     neighbors_embeddings, _ = self.selfAttention(neighbors_embeddings, neighbors_embeddings,
        #                                                  neighbors_embeddings)
        #
        #     side_embeddings.data[key] = neighbors_embeddings[0]

        sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
        bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))

        embeddings = sum_embeddings + bi_embeddings
        return embeddings


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class UIR_KG(nn.Module):
    def __init__(self, args, n_users, n_entities, n_relations, train_user_dict, train_user_dict2,A_in=None, user_pre_embed=None, item_pre_embed=None):
        super(UIR_KG, self).__init__()

        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.train_user_dict = train_user_dict
        self.train_user_dict2 = train_user_dict2

        self.embed_dim = args.embedding_dim  # 64
        self.relation_dim = args.relation_dim  # 64

        self.conv_dim_list = [args.embedding_dim] + eval(args.conv_dim_list)  # [64,64,32,16]
        self.mess_dropout = eval(args.mess_dropout)  # [0.1,0.1,0.1]
        self.n_layers = len(eval(args.conv_dim_list))  # 3

        self.kg_l2loss_lambda = args.kg_l2loss_lambda  # 1e-5
        self.cf_l2loss_lambda = args.cf_l2loss_lambda  # 1e-5

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(
                torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))  # 初始化实体节点嵌入
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)  # 初始化关系嵌入列表
        nn.init.xavier_uniform_(self.trans_M)  # 初始化论文中的 W_r

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k]))

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

    def calc_cf_embeddings(self):
        ego_embeddings = self.entity_user_embed.weight
        all_embeddings = [ego_embeddings]
        for idx, layer in enumerate(self.aggregator_layers):
            ego_embeddings = layer(ego_embeddings, self.A_in, self.train_user_dict2)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=1)
        return all_embeddings

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        # len(user_ids) == len(item_pos_ids) == len(item_neg_ids) == cf_batch_size
        all_embeddings = self.calc_cf_embeddings()
        user_embeddings = all_embeddings[user_ids]
        item_pos_embeddings = all_embeddings[item_pos_ids]
        item_neg_embeddings = all_embeddings[item_neg_ids]

        # 对应位置元素相乘，然后行相加
        pos_score = torch.sum(user_embeddings * item_pos_embeddings, dim=1)
        neg_score = torch.sum(user_embeddings * item_neg_embeddings, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embeddings) + _L2_loss_mean(item_pos_embeddings) + _L2_loss_mean(
            item_neg_embeddings)

        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        # len(h) == len(r) == len(pos_t) == len(neg_t) == kg_batch_size
        r_embeddings = self.relation_embed(r)
        W_r = self.trans_M[r]

        h_embeddings = self.entity_user_embed(h)
        pos_t_embeddings = self.entity_user_embed(pos_t)
        neg_t_embeddings = self.entity_user_embed(neg_t)

        # .unsqueeze()升维操作
        r_mul_h = torch.bmm(h_embeddings.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embeddings.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embeddings.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        pos_score = torch.sum(torch.pow(r_mul_h + r_embeddings - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embeddings - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embeddings) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss

        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embeddings = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embeddings = self.entity_user_embed[h_list]
        t_embeddings = self.entity_user_embed[t_list]

        r_mul_h = torch.matmul(h_embeddings, W_r)
        r_mul_t = torch.matmul(t_embeddings, W_r)

        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embeddings), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(self.A_in.shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def cal_score(self, user_ids, item_ids):
        all_embeddings = self.calc_cf_embeddings()
        user_embeddings = all_embeddings[user_ids]
        item_embeddings = all_embeddings[item_ids]

        # cf_score.shape is (len(batch_user_ids), n_items)
        cf_score = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        return cf_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'updata_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.cal_score(*input)
