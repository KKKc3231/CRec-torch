import sys
import csv
import copy
import torch
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import torch

def build_index(dataset_name):

    ui_mat = np.loadtxt('/home/jovyan/Recommend-develop_zyf_demo_v3/rcar_cartpye_rank/CRec_torch/seq_model/data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    
    # 提前构造好用户序列和物品序列的列表
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

def build_index_pd(fname): 
    csv_path = '/data/cc/GR_recommand/SeqRec/data/%s.csv' % fname
    user_item_df = pd.read_csv(csv_path)
    user_col = user_item_df['user_map_guid'].astype(int).values
    item_col = user_item_df['item_map_id'].astype(int).values
    ui_mat = np.column_stack((user_col, item_col))
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    
    # 提前构造好用户序列和物品序列的列表
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
# 随机负采样，不包含重复的
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# 采样函数，采样正负样本
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        # 如果当前user的序列长度<=1, 序列太短了，则随机再采样一个用户id
        while len(user_train[uid]) <= 1: 
            uid = np.random.randint(1, usernum + 1)

        # 用户的最长序列为200
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        # nxt：用户最后一个交互的物品，即需要预测的 
        nxt = user_train[uid][-1]
        idx = maxlen - 1
        
        # set一下，去重复, 也可以理解为带时间戳的
        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # 随机负采样
            if nxt != 0: 
                # 负样本不在序列内
                neg[idx] = random_neq(1, itemnum + 1, ts)

            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            # 随机打散
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# data_split_pd
# 是按用户划分的吗？
def data_partition_pd(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    csv_path = '/data/cc/GR_recommand/SeqRec/data/%s.csv' % fname
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            u = line[0]  # 用户 ID
            i = line[1]  # 物品 ID
            
            # 转换为整数
            u = int(u)
            i = int(i)
            
            # 更新用户数和物品数的最大值
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            
            # 更新用户-物品字典
            if u not in User:
                User[u] = []
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    # assume user/item index starting from 1
    f = open('/home/jovyan/Recommend-develop_zyf_demo_v3/rcar_cartpye_rank/CRec_torch/seq_model/data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, k, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        # 取出当前需要推理的作为输入的最后一个
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        # 训练集的用户序列，可以得到
        rated = set(train[u])

        # 为啥要加0到最开始呢？
        rated.add(0)
        item_idx = [test[u][0]]
        
        # 可以理解为召回的其他100个候选物品
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            # 下面的逻辑是不和训练集合重复，但是租车的话，这是点击序列，之后还可能再点击
            #while t in rated: 
            #    t = np.random.randint(1, itemnum + 1)
            item_idx.append(t) # 101个候选

        # 得到的是模型的输入，userid，seq：长度为200，item_idx: 候选集
        # 因为小的要排前面，所以要加一个负号
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        # item_idx中第一个是用户真实交互过的
        rank = predictions.argsort().argsort()[0].item()

        # 对应索引
        index_rank1 = torch.where(predictions.argsort().argsort() == 1)[0]

        # recommand
        real_item_id = item_idx[0]
        predict_item_id = item_idx[index_rank1]


        valid_user += 1

        # ndcg@10 && hr@10
        if rank <= k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    print("user_id", u)
    print("real_item_id:", real_item_id)
    print("recommand_item_id", predict_item_id)

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, k, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        # item_idx中第一个是用户真实交互过的
        rank = predictions.argsort().argsort()[0].item()
        
        valid_user += 1

        if rank <= k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
