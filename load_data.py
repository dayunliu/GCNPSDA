import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        # print(path)
        self.batch_size = batch_size

        train_file = 'train_2.txt'
        test_file = 'test_2.txt'

        self.n_snoRNAs, self.n_diseases = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_snoRNAs = []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    diseases = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_snoRNAs.append(uid)
                    self.n_diseases = max(self.n_diseases, max(diseases))
                    self.n_snoRNAs = max(self.n_snoRNAs, uid)
                    self.n_train += len(diseases)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        diseases = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_diseases = max(self.n_diseases, max(diseases))
                    self.n_test += len(diseases)
        self.n_diseases += 1
        self.n_snoRNAs += 1
        # self.print_statistics()
        self.R = sp.dok_matrix((self.n_snoRNAs, self.n_diseases), dtype=np.float32)
        self.train_diseases, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    diseases = [int(i) for i in l.split(' ')]
                    uid, train_diseases = diseases[0], diseases[1:]

                    for i in train_diseases:
                        self.R[uid, i] = 1.
                        
                    self.train_diseases[uid] = train_diseases
                    
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        diseases = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_diseases = diseases[0], diseases[1:]
                    self.test_set[uid] = test_diseases

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz('s_adj_mat.npz')
            norm_adj_mat = sp.load_npz('s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz('s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz('s_adj_mat.npz', adj_mat)
            sp.save_npz('s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz('s_mean_adj_mat.npz', mean_adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz('s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz('s_pre_adj_mat.npz', norm_adj)
            
        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_snoRNAs + self.n_diseases, self.n_snoRNAs + self.n_diseases), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_snoRNAs*i/5.0):int(self.n_snoRNAs*(i+1.0)/5), self.n_snoRNAs:] =\
            R[int(self.n_snoRNAs*i/5.0):int(self.n_snoRNAs*(i+1.0)/5)]
            adj_mat[self.n_snoRNAs:,int(self.n_snoRNAs*i/5.0):int(self.n_snoRNAs*(i+1.0)/5)] =\
            R[int(self.n_snoRNAs*i/5.0):int(self.n_snoRNAs*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        
    def negative_pool(self):
        t1 = time()
        for u in self.train_diseases.keys():
            neg_diseases = list(set(range(self.n_diseases)) - set(self.train_diseases[u]))
            pools = [rd.choice(neg_diseases) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_snoRNAs:
            snoRNAs = rd.sample(self.exist_snoRNAs, self.batch_size)
        else:
            snoRNAs = [rd.choice(self.exist_snoRNAs) for _ in range(self.batch_size)]


        def sample_pos_diseases_for_u(u, num):
            pos_diseases = self.train_diseases[u]
            n_pos_diseases = len(pos_diseases)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_diseases, size=1)[0]
                pos_i_id = pos_diseases[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_diseases_for_u(u, num):
            neg_diseases = []
            while True:
                if len(neg_diseases) == num: break
                neg_id = np.random.randint(low=0, high=self.n_diseases,size=1)[0]
                if neg_id not in self.train_diseases[u] and neg_id not in neg_diseases:
                    neg_diseases.append(neg_id)
            return neg_diseases

        def sample_neg_diseases_for_u_from_pools(u, num):
            neg_diseases = list(set(self.neg_pools[u]) - set(self.train_diseases[u]))
            return rd.sample(neg_diseases, num)

        pos_diseases, neg_diseases = [], []
        for u in snoRNAs:
            pos_diseases += sample_pos_diseases_for_u(u, 1)
            neg_diseases += sample_neg_diseases_for_u(u, 1)

        return snoRNAs, pos_diseases, neg_diseases
    
    def sample_test(self):
        if self.batch_size <= self.n_snoRNAs:
            snoRNAs = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            snoRNAs = [rd.choice(self.exist_snoRNAs) for _ in range(self.batch_size)]

        def sample_pos_diseases_for_u(u, num):
            pos_diseases = self.test_set[u]
            n_pos_diseases = len(pos_diseases)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_diseases, size=1)[0]
                pos_i_id = pos_diseases[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_diseases_for_u(u, num):
            neg_diseases = []
            while True:
                if len(neg_diseases) == num: break
                neg_id = np.random.randint(low=0, high=self.n_diseases, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_diseases[u]) and neg_id not in neg_diseases:
                    neg_diseases.append(neg_id)
            return neg_diseases
    
        def sample_neg_diseases_for_u_from_pools(u, num):
            neg_diseases = list(set(self.neg_pools[u]) - set(self.train_diseases[u]))
            return rd.sample(neg_diseases, num)

        pos_diseases, neg_diseases = [], []
        for u in snoRNAs:
            pos_diseases += sample_pos_diseases_for_u(u, 1)
            neg_diseases += sample_neg_diseases_for_u(u, 1)

        return snoRNAs, pos_diseases, neg_diseases
    
    
    
    
    
    
    def get_num_snoRNAs_diseases(self):
        return self.n_snoRNAs, self.n_diseases

    # # def print_statistics(self):
    #     print('n_snoRNAs=%d, n_diseases=%d' % (self.n_snoRNAs, self.n_diseases))
    #     print('n_interactions=%d' % (self.n_train + self.n_test))
    #     print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_snoRNAs * self.n_diseases)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open('sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open('sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_snoRNAs_to_test = list(self.test_set.keys())
        snoRNA_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_snoRNAs_to_test:
            train_iids = self.train_diseases[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in snoRNA_n_iid.keys():
                snoRNA_n_iid[n_iids] = [uid]
            else:
                snoRNA_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole snoRNA set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(snoRNA_n_iid)):
            temp += snoRNA_n_iid[n_iids]
            n_rates += n_iids * len(snoRNA_n_iid[n_iids])
            n_count -= n_iids * len(snoRNA_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per snoRNA<=[%d], #snoRNAs=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(snoRNA_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per snoRNA<=[%d], #snoRNAs=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
