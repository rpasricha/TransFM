# PRME-FM implementation
import pandas as pd
import scipy.sparse as sp
import random
import numpy as np
import tensorflow as tf
import dataset
import sys

class PRME_FM:
    def __init__(self, dataset, args):
        print 'In class PRME_FM'
        self.dataset = dataset
        self.args = args

        # Use a training batch to figure out feature dimensionality
        users, pos_feats, neg_feats = self.dataset.generate_train_batch_sp()
        self.feature_dim = pos_feats.shape[1]
        print 'Feature dimension = ' + str(self.feature_dim)

    def get_preds(self, var_linear, var_emb_factors,
            sparse_pos_feats, sparse_neg_feats):

        # Linear terms
        pos_linear = tf.sparse_tensor_dense_matmul(sparse_pos_feats, var_linear)
        neg_linear = tf.sparse_tensor_dense_matmul(sparse_neg_feats, var_linear)

        # Interaction terms
        # First define common terms that are used by future calculations
        # Common terms
        var_emb_product = tf.reduce_sum(tf.square(var_emb_factors), axis=1, keep_dims=True)

        # Common terms positive
        pos_feats_sum = tf.sparse_reduce_sum(sparse_pos_feats, axis=1, keep_dims=True)
        pos_emb_mul = tf.sparse_tensor_dense_matmul(sparse_pos_feats, var_emb_factors)

        # Common terms negative
        neg_feats_sum = tf.sparse_reduce_sum(sparse_neg_feats, axis=1, keep_dims=True)
        neg_emb_mul = tf.sparse_tensor_dense_matmul(sparse_neg_feats, var_emb_factors)

        # Term 1 pos
        prod_term_pos = tf.sparse_tensor_dense_matmul(
                sparse_pos_feats, var_emb_product)
        term_1_pos = prod_term_pos * pos_feats_sum

        # Term 1 neg
        prod_term_neg = tf.sparse_tensor_dense_matmul(
                sparse_neg_feats, var_emb_product)
        term_1_neg = prod_term_neg * neg_feats_sum

        # Term 2
        term_2_pos = 2 * tf.reduce_sum(tf.square(pos_emb_mul), axis=1, keep_dims=True)
        term_2_neg = 2 * tf.reduce_sum(tf.square(neg_emb_mul), axis=1, keep_dims=True)

        # Term 3
        term_3_pos = term_1_pos
        term_3_neg = term_1_neg

        # Predictions
        pos_preds = pos_linear + 0.5 * (term_1_pos - term_2_pos + term_3_pos)
        neg_preds = neg_linear + 0.5 * (term_1_neg - term_2_neg + term_3_neg)

        return pos_preds, neg_preds

    def create_model(self):
        g = tf.Graph()
        with g.as_default():
            # Define model variables
            var_linear = tf.get_variable('linear',
                    [self.feature_dim, 1],
                    initializer=tf.random_uniform_initializer(
                        -self.args.init_mean, self.args.init_mean))

            var_emb_factors = tf.get_variable('emb_factors',
                    [self.feature_dim, self.args.num_dims],
                    initializer=tf.random_uniform_initializer(
                        -self.args.init_mean, self.args.init_mean))

            # Sparse placeholders
            pl_user_list = tf.placeholder(tf.int64, shape=[None], name='pos_list')

            pl_pos_indices = tf.placeholder(tf.int64, shape=[None, 2], name='pos_indices')
            pl_pos_values = tf.placeholder(tf.float32, shape=[None], name='pos_values')
            pl_pos_shape = tf.placeholder(tf.int64, shape=[2], name='pos_shape')

            pl_neg_indices = tf.placeholder(tf.int64, shape=[None, 2], name='neg_indices')
            pl_neg_values = tf.placeholder(tf.float32, shape=[None], name='neg_values')
            pl_neg_shape = tf.placeholder(tf.int64, shape=[2], name='neg_shape')

            placeholders = {
                    'pl_user_list': pl_user_list,
                    'pl_pos_indices': pl_pos_indices,
                    'pl_pos_values': pl_pos_values,
                    'pl_pos_shape': pl_pos_shape,
                    'pl_neg_indices': pl_neg_indices,
                    'pl_neg_values': pl_neg_values,
                    'pl_neg_shape': pl_neg_shape
            }

            # Input positive features, shape = (batch_size * feature_dim)
            sparse_pos_feats = tf.SparseTensor(pl_pos_indices, pl_pos_values, pl_pos_shape)

            # Input negative features, shape = (batch_size * feature_dim)
            sparse_neg_feats = tf.SparseTensor(pl_neg_indices, pl_neg_values, pl_neg_shape)

            pos_preds, neg_preds = self.get_preds(var_linear, var_emb_factors,
                    sparse_pos_feats, sparse_neg_feats)

            l2_reg = tf.add_n([
                self.args.linear_reg * tf.reduce_sum(tf.square(var_linear)),
                self.args.emb_reg    * tf.reduce_sum(tf.square(var_emb_factors)),
            ])

            # BPR training op (add 1e-10 to help numerical stability)
            bprloss_op = tf.reduce_sum(tf.log(1e-10 + tf.sigmoid(pos_preds - neg_preds))) - l2_reg
            bprloss_op = -bprloss_op

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.args.starting_lr,
                global_step, self.args.lr_decay_freq,
                self.args.lr_decay_factor, staircase=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(bprloss_op, global_step=global_step)

            # AUC
            binary_ranks = tf.to_float((pos_preds - neg_preds) > 0)
            auc_per_user = tf.segment_mean(binary_ranks, pl_user_list)
            auc_op = tf.divide(tf.reduce_sum(auc_per_user),
                    tf.to_float(tf.size(tf.unique(pl_user_list)[0])))

        self.var_linear = var_linear
        self.var_emb_factors = var_emb_factors
        return (g, bprloss_op, optimizer, train_op, auc_op, l2_reg, placeholders)

    def create_feed_dict(self, placeholders, users, pos_feats, neg_feats):
        feed_dict = {
                placeholders['pl_user_list']: users.nonzero()[1],
                placeholders['pl_pos_indices']: np.hstack((
                    pos_feats.nonzero()[0][:, None],
                    pos_feats.nonzero()[1][:, None],
                )),
                placeholders['pl_pos_values']: pos_feats.data,
                placeholders['pl_pos_shape']: pos_feats.shape,
                placeholders['pl_neg_indices']: np.hstack((
                    neg_feats.nonzero()[0][:, None],
                    neg_feats.nonzero()[1][:, None],
                )),
                placeholders['pl_neg_values']: neg_feats.data,
                placeholders['pl_neg_shape']: neg_feats.shape,
        }

        return feed_dict

    def train(self):
        (g, bprloss_op, optimizer, train_op, auc_op, l2_reg,
                placeholders) = self.create_model()

        with g.as_default():
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            best_epoch = 0
            best_val_auc = -1
            best_test_auc = -1

            for epoch in xrange(self.args.max_iters):
                print 'Epoch: {}'.format(epoch),
                users, pos_feats, neg_feats = self.dataset.generate_train_batch_sp()
                feed_dict = self.create_feed_dict(placeholders, users, pos_feats, neg_feats)
                loss, train_auc, l2, lr, _ = sess.run(
                    [bprloss_op, auc_op, l2_reg, optimizer._lr, train_op],
                    feed_dict = feed_dict)
                print '\tLoss = {}'.format(loss)

                if epoch % self.args.eval_freq == 0:
                    users, pos_feats, neg_feats = self.dataset.generate_val_batch_sp()
                    feed_dict = self.create_feed_dict(placeholders, users, pos_feats, neg_feats)
                    val_auc = sess.run(auc_op, feed_dict=feed_dict)

                    users, pos_feats, neg_feats = self.dataset.generate_test_batch_sp()
                    feed_dict = self.create_feed_dict(placeholders, users, pos_feats, neg_feats)
                    test_auc = sess.run(auc_op, feed_dict = feed_dict)

                    print '\tVal AUC = ' + str(val_auc) + '\tTest AUC = ' + str(test_auc)

                    if val_auc > best_val_auc:
                        best_epoch = epoch
                        best_val_auc = val_auc
                        best_test_auc = test_auc
                    else:
                        if epoch >= (best_epoch + self.args.quit_delta):
                            print 'Overfitted, exiting...'
                            print '\tBest Epoch = {}'.format(best_epoch)
                            print '\tValidation AUC = {}'.format(best_val_auc)
                            print '\tTest AUC = {}'.format(best_test_auc)
                            break

                    print '\tCurrent max = {} at epoch {}'.format(
                            best_val_auc, best_epoch)

        return (best_val_auc, best_test_auc)

