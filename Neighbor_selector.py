import numpy as np
import random

num_path = 1500
path_len = 5
sample_num_path = 5

file_path = './datasets/last-fm'
save_path = './data_processing/last-fm'
train_file = file_path + '/train.txt'
test_file = file_path + '/test.txt'
kg_file = file_path + '/kg_final.txt'

# file = open(file_path + 'CKG_Path_Sample(num_sample_'+ str(num_path) + ').txt','a')

# file = open(file_path + '/sample_num_path/CKG_Path_Sample(sample_num_path_'+ str(sample_num_path) + ').txt','a')
# file = open(save_path + '/path_len/CKG_Path_Sample(path_len_'+ str(path_len) + ').txt','a')

file = open(save_path + '/Neighbor_selector.txt','a')

def _load_ratings(file_name):
    user_dict = dict()
    inter_mat = list()

    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

        if len(pos_ids) > 0:
            user_dict[u_id] = pos_ids
    return np.array(inter_mat), user_dict



train_data, train_user_dict = _load_ratings(train_file)


test_data, test_user_dict = _load_ratings(test_file)
exist_users = train_user_dict.keys()

n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1 # 45919
n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1 # 45538
n_train = len(train_data)
n_test = len(test_data)

def _get_item_dict(train_data):
    item_dict = dict()
    for data in train_data:
        if data[1] not in item_dict.keys():
            item_dict[data[1]] = [data[0]]
        else:
            l = list(item_dict[data[1]])
            l.append(data[0])
            item_dict[data[1]] = l
    return item_dict

def _load_kg(file_name):
    kg_dict = {}
    reverse_kg_dict = {}
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]
        h_id, r_id, t_id = inters[0], inters[1], inters[2]
        # if h_id == 45545:
        #     print(r_id,t_id)

        if h_id not in kg_dict.keys():
            kg_dict[h_id] = [t_id]
        else:
            l = kg_dict[h_id]
            l.append(t_id)
            kg_dict[h_id] = l

        if t_id not in kg_dict.keys():
            kg_dict[t_id] = [h_id]
        else:
            l = kg_dict[t_id]
            l.append(h_id)
            kg_dict[t_id] = l

        if t_id not in reverse_kg_dict.keys():
            reverse_kg_dict[t_id] = [h_id]
        else:
            l = reverse_kg_dict[t_id]
            l.append(h_id)
            reverse_kg_dict[t_id] = l

    return kg_dict,reverse_kg_dict

def _get_neighbor(id,dict):
    neighbor_id_list = dict[id]
    idx = random.randint(0,len(neighbor_id_list)-1)
    return neighbor_id_list[idx]


train_item_dict = _get_item_dict(train_data)


#kg_dict value 最小长度为5
kg_dict, reverse_kg_dict = _load_kg(kg_file)#[n_entities, n_relations, n_triples]=[136499, 42, 1853704]



n_entities = len(kg_dict.keys())#136499



# CKG随机采样协同路径
path_dict = dict()
for target_user_id in exist_users:
    path_list = []

    for i in range(num_path):
        path = []
        path.append(_get_neighbor(target_user_id, train_user_dict))
        for j in range(path_len-1):
            id = path[len(path) - 1]
            entity_id = _get_neighbor(id,kg_dict)
            path.append(entity_id)
        if path not in path_list:
            path_list.append(path)

    #     print(i)
    path_dict[target_user_id] = path_list

print("path sample down!")


# # a = 0

for key in path_dict.keys():
    path_list = path_dict[key]
    item_num_list = [] # Record the number of items in each path
    mean_popularity_list = [] # Record average item popularity for each path
    lower_popularity_list = []  # Record the paths with the lowest average item popularity
    neighbors = []
    _str = ""

    for p in path_list:#p is a collaborative path
        popularity = 0
        n = 0
        for entity_id in p:
            if entity_id < n_items:
                n += 1
                popularity += len(train_item_dict[entity_id])
        item_num_list.append(n)
        mean_popularity_list.append(popularity/n)
    popularity_array = np.array(mean_popularity_list)
    idx_array = np.argsort(popularity_array)

    for i in range(sample_num_path):
        lower_popularity_list.append(path_list[idx_array[i]])
    _str =_str + str(key) + " "

    for p in lower_popularity_list:
        for entity_id in p:
            if entity_id < n_items and entity_id not in neighbors:
                neighbors.append(entity_id)
                _str = _str + str(entity_id) + " "

    # _str += "mean_pop is "
    # for i in range(sample_num_path):
    #     _str = _str + str(mean_popularity_list[idx_array[i]]) + " "
    # _str += "items_number is "
    # for i in range(sample_num_path):
    #     _str = _str + str(item_num_list[idx_array[i]]) + " "
    # _str += "neigbors_num is " + str(len(neighbors))


    file.write(_str + "\n")







