
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import argparse
from load_data import *


class IMP_GCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'GCNPSDA'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.group = args.groups

        self.pretrain_data = pretrain_data

        self.n_snoRNAs = data_config['n_snoRNAs']
        self.n_diseases = data_config['n_diseases']

        self.n_fold = 20

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.dropout = args.mlp_dropout

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose
        # placeholder definition
        self.snoRNAs = tf.placeholder(tf.int32, shape=(None,))
        self.pos_diseases = tf.placeholder(tf.int32, shape=(None,))
        self.neg_diseases = tf.placeholder(tf.int32, shape=(None,))
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        self.weights = self._init_weights()

        
        self.ua_embeddings, self.ia_embeddings, self.A_fold_hat_group_filter, self.snoRNA_group_embeddings = self._create_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.snoRNAs)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_diseases)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_diseases)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['snoRNA_embedding'], self.snoRNAs)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['disease_embedding'], self.pos_diseases)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['disease_embedding'], self.neg_diseases)

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['snoRNA_embedding'] = tf.Variable(initializer([self.n_snoRNAs, self.emb_dim]),
                                                        name='snoRNA_embedding')
            all_weights['disease_embedding'] = tf.Variable(initializer([self.n_diseases, self.emb_dim]),
                                                        name='disease_embedding')
            print('using xavier initialization')
        else:
            all_weights['snoRNA_embedding'] = tf.Variable(initial_value=self.pretrain_data['snoRNA_embeded'], trainable=True,
                                                        name='snoRNA_embedding', dtype=tf.float32)
            all_weights['disease_embedding'] = tf.Variable(initial_value=self.pretrain_data['disease_embeded'], trainable=True,
                                                        name='disease_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['W_gc_1'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='W_gc_1')
        all_weights['b_gc_1'] = tf.Variable(initializer([1, self.emb_dim]), name='b_gc_1')

        all_weights['W_gc_2'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='W_gc_2')
        all_weights['b_gc_2'] = tf.Variable(initializer([1, self.emb_dim]), name='b_gc_2')

        all_weights['W_gc'] = tf.Variable(initializer([self.emb_dim, self.group]), name='W_gc')
        all_weights['b_gc'] = tf.Variable(initializer([1, self.group]), name='b_gc')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_snoRNAs + self.n_diseases) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_snoRNAs + self.n_diseases
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = tf.transpose(group_embedding)
        A_fold_hat_group = []
        A_fold_hat_group_filter = []
        A_fold_hat = []

        # split L in fold
        fold_len = (self.n_snoRNAs + self.n_diseases) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_snoRNAs + self.n_diseases
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))

        # k groups
        for k in range(0, self.group):
            A_fold_disease_filter = []
            A_fold_hat_disease = []

            # n folds in per group (filter snoRNA)
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = self.n_snoRNAs + self.n_diseases
                else:
                    end = (i_fold + 1) * fold_len

                A_fold_hat_disease.append(A_fold_hat[i_fold].__mul__(group_embedding[k]).__mul__(
                    tf.expand_dims(group_embedding[k][start:end], axis=1)))
                disease_filter = tf.sparse_reduce_sum(A_fold_hat_disease[i_fold], axis=1)
                disease_filter = tf.where(disease_filter > 0., x=tf.ones_like(disease_filter), y=tf.zeros_like(disease_filter))
                A_fold_disease_filter.append(disease_filter)

            A_fold_disease = tf.concat(A_fold_disease_filter, axis=0)
            A_fold_hat_group_filter.append(A_fold_disease)
            A_fold_hat_group.append(A_fold_hat_disease)

        return A_fold_hat_group, A_fold_hat_group_filter


    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_snoRNAs + self.n_diseases) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_snoRNAs + self.n_diseases
            else:
                end = (i_fold + 1) * fold_len
            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        return A_fold_hat

    def _create_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['snoRNA_embedding'], self.weights['disease_embedding']], axis=0)

        # group snoRNAs
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
        snoRNA_group_embeddings_side = tf.concat(temp_embed, 0) + ego_embeddings


        snoRNA_group_embeddings_hidden_1 = tf.nn.leaky_relu(tf.matmul(snoRNA_group_embeddings_side, self.weights['W_gc_1']) + self.weights['b_gc_1'])

        snoRNA_group_embeddings_hidden_d1 = tf.nn.dropout(snoRNA_group_embeddings_hidden_1, 0.6)
        snoRNA_group_embeddings_sum = tf.matmul(snoRNA_group_embeddings_hidden_d1, self.weights['W_gc']) + self.weights['b_gc']

        # snoRNA 0-1
        a_top, a_top_idx = tf.nn.top_k(snoRNA_group_embeddings_sum, 1, sorted=False)
        snoRNA_group_embeddings = tf.cast(tf.equal(snoRNA_group_embeddings_sum,a_top), dtype=tf.float32)
        u_group_embeddings, i_group_embeddings = tf.split(snoRNA_group_embeddings, [self.n_snoRNAs, self.n_diseases], 0)
        i_group_embeddings = tf.ones(tf.shape(i_group_embeddings))
        snoRNA_group_embeddings = tf.concat([u_group_embeddings, i_group_embeddings], axis = 0)
        # Matrix mask
        A_fold_hat_group, A_fold_hat_group_filter = self._split_A_hat_group(self.norm_adj, snoRNA_group_embeddings)

        # embedding transformation
        all_embeddings = [ego_embeddings]
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

        side_embeddings = tf.concat(temp_embed, 0)
        all_embeddings += [side_embeddings]

        ego_embeddings_g = []
        for g in range(0,self.group):
            ego_embeddings_g.append(ego_embeddings)

        ego_embeddings_f = []
        for k in range(1, self.n_layers):
            for g in range(0,self.group):
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat_group[g][f], ego_embeddings_g[g]))
                side_embeddings = tf.concat(temp_embed, 0)
                ego_embeddings_g[g]=ego_embeddings_g[g] + side_embeddings
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], side_embeddings))
                if k == 1:
                    ego_embeddings_f.append(tf.concat(temp_embed, 0))
                else:
                    ego_embeddings_f[g] = tf.concat(temp_embed, 0)
            ego_embeddings = tf.reduce_sum(ego_embeddings_f, axis=0, keepdims=False)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_snoRNAs, self.n_diseases], 0)
        return u_g_embeddings, i_g_embeddings, A_fold_hat_group_filter, snoRNA_group_embeddings_sum




    def create_bpr_loss(self, snoRNAs, pos_diseases, neg_diseases):
        pos_scores = tf.reduce_sum(tf.multiply(snoRNAs, pos_diseases), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(snoRNAs, neg_diseases), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)
        return pre_out * tf.div(1., keep_prob)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GCNPSDA.")
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, home_kitchen, KS10}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--groups', type=int, default=2,
                        help='Number of group.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='IMP_GCN',
                        help='Specify the type of the graph convolutional layer from {IMP_GCN, LightGCN}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--mlp_dropout', nargs='?', default='1.0',
                        help='Keep probability of MLP layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = dict()
    config['n_snoRNAs'] = data_generator.n_snoRNAs
    config['n_diseases'] = data_generator.n_diseases
    plain_adj, norm_adj, mean_adj,pre_adj = data_generator.get_adj_mat()
    config['norm_adj'] = pre_adj

    pretrain_data = None

    model = IMP_GCN(data_config=config, pretrain_data=pretrain_data)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    """
    Train.
    """

    for epoch in range(10):
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            snoRNAs, pos_diseases, neg_diseases = data_generator.sample()
            _, batch_loss = sess.run([model.opt, model.loss],
                               feed_dict={model.snoRNAs: snoRNAs, model.pos_diseases: pos_diseases,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_diseases: neg_diseases})
            loss += batch_loss/n_batch


