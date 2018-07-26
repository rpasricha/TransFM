import pandas as pd
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import copy
import os

# Class to represent a dataset
class Dataset:
    def __init__(self, path, args, user_min=5, item_min=5):
        self.user_min = user_min
        self.item_min = item_min
        self.args = args

        df = pd.read_csv(path, sep=' ', header=None,
                names=['user_id', 'item_id', 'rating', 'time'], index_col=False)

        print 'First pass'
        print '\tnum_users = ' + str(len(df['user_id'].unique()))
        print '\tnum_items = ' + str(len(df['item_id'].unique()))
        print '\tdf_shape  = ' + str(df.shape)

        user_counts = df['user_id'].value_counts()
        print 'Collected user counts...'
        item_counts = df['item_id'].value_counts()
        print 'Collected item counts...'

        # Filter based on user and item counts
        df = df[df.apply(
            lambda x: user_counts[x['user_id']] >= user_min, axis=1)]
        print 'User filtering done...'
        df = df[df.apply(
            lambda x: item_counts[x['item_id']] >= item_min, axis=1)]
        print 'Item filtering done...'

        print 'Second pass'
        print '\tnum_users = ' + str(len(df['user_id'].unique()))
        print '\tnum_items = ' + str(len(df['item_id'].unique()))
        print '\tdf_shape  = ' + str(df.shape)

        # Normalize temporal values
        print 'Normalizing temporal values...'
        mean = df['time'].mean()
        std  = df['time'].std()
        self.ONE_YEAR = (60 * 60 * 24 * 365) / mean
        self.ONE_DAY = (60 * 60 * 24) / mean
        df['time'] = (df['time'] - mean) / std

        print 'Constructing datasets...'
        training_set = defaultdict(list)
        # Start counting users and items at 1 to facilitate sparse matrix
        # computation.
        num_users = 1
        num_items = 1
        item_to_idx = {}
        user_to_idx = {}
        idx_to_item = {}
        idx_to_user = {}
        for row in df.itertuples():
            # New item
            if row.item_id not in item_to_idx:
                item_to_idx[row.item_id] = num_items
                idx_to_item[num_items] = row.item_id
                num_items += 1

            # New user
            if row.user_id not in user_to_idx:
                user_to_idx[row.user_id] = num_users
                idx_to_user[num_users] = row.user_id
                num_users += 1

            # Converts all ratings to positive implicit feedback
            training_set[user_to_idx[row.user_id]].append(
                    (item_to_idx[row.item_id], row.time))

        for user in training_set:
            training_set[user].sort(key=lambda x: x[1])

        training_times = {}
        val_set = {}
        val_times = {}
        test_set = {}
        test_times = {}
        # Map from user to set of items for easy lookup
        item_set_per_user = {}
        for user in training_set:
            if len(training_set[user]) < 3:
                # Reviewed < 3 items, insert dummy values
                test_set[user] = (-1, -1)
                test_times[user] = (-1, -1)
                val_set[user] = (-1, -1)
                val_times[user] = (-1, -1)
            else:
                test_item, test_time = training_set[user].pop()
                val_item, val_time = training_set[user].pop()
                last_item, last_time = training_set[user][-1]
                test_set[user] = (test_item, val_item)
                test_times[user] = (test_time, val_time)
                val_set[user] = (val_item, last_item)
                val_times[user] = (val_time, last_time)

            # Separate timestamps and create item set
            training_times[user] = copy.deepcopy(training_set[user])
            training_set[user] = map(lambda x: x[0], training_set[user])
            item_set_per_user[user] = set(training_set[user])

        num_train_events = 0
        for user in training_set:
            num_train_events += len(training_set[user])

        self.training_set = training_set
        self.training_times = training_times
        self.val_set = val_set
        self.val_times = val_times
        self.test_set = test_set
        self.test_times = test_times
        self.item_set_per_user = item_set_per_user

        self.item_to_idx = item_to_idx
        self.user_to_idx = user_to_idx
        self.idx_to_item = idx_to_item
        self.idx_to_user = idx_to_user

        self.num_users = num_users
        self.num_items = num_items
        self.num_train_events = num_train_events

        # Read item categories
        if self.args.features == 'categories':
            cat_seq_df = pd.read_csv(self.args.features_file)
            cat_seq_df['item_cat_seq'] = cat_seq_df['item_cat_seq'].apply(eval)
            cat_rows = []
            cat_cols = []
            cat_data = []
            for row in cat_seq_df.itertuples():
                if row.item_id not in self.item_to_idx:
                    continue
                # Subtract to account for no item with index 0
                item_idx = self.item_to_idx[row.item_id] - 1
                for item in row.item_cat_seq:
                    cat_rows.append(item_idx)
                    cat_cols.append(item)
                    cat_data.append(1)
            self.cat_mat = sp.coo_matrix((cat_data, (cat_rows, cat_cols))).tocsr()
        else:
            self.cat_mat = None

        # Read user/item content info
        if self.args.features == 'content':
            print 'Reading user demographics...'
            user_df = pd.read_csv(self.args.features_file.split(',')[0])
            user_df = user_df.set_index('idx')
            self.user_df = user_df

            self.orig_indices = []
            for i in range(1, self.num_users):
                self.orig_indices.append(self.idx_to_user[i])
            self.user_feats = sp.csr_matrix(user_df.loc[self.orig_indices].values)

            print 'Reading item demographics...'
            item_df = pd.read_csv(self.args.features_file.split(',')[1])
            item_df = item_df.set_index('idx')
            self.item_df = item_df

            self.orig_item_indices = []
            for i in range(1, self.num_items):
                self.orig_item_indices.append(self.idx_to_item[i])
            self.item_feats = sp.csr_matrix(item_df.loc[self.orig_item_indices].values)
        else:
            self.user_feats = None
            self.item_feats = None

        # Read geographical content info
        if self.args.features == 'geo':
            print 'Reading geographical features...'

            geo_df = pd.read_csv(self.args.features_file)
            geo_df = geo_df.set_index('place_id')

            self.orig_item_indices = []
            for i in range(1, self.num_items):
                self.orig_item_indices.append(self.idx_to_item[i])

            self.geo_feats = sp.csr_matrix(geo_df.loc[self.orig_item_indices].values)
        else:
            self.geo_feats = None

        # Create scipy.sparse matrices
        self.user_one_hot = sp.identity(self.num_users - 1).tocsr()
        self.item_one_hot = sp.identity(self.num_items - 1).tocsr()

        # Sparse training matrices
        train_rows = []
        train_cols = []
        train_vals = []
        train_prev_vals = []
        train_times = []
        train_prev_times = []
        for user in self.training_set:
            for i in range(1, len(self.training_set[user])):
                item = self.training_set[user][i]
                item_prev = self.training_set[user][i-1]
                item_time = self.training_times[user][i]
                item_prev_time = self.training_times[user][i-1]
                train_rows.append(user)
                train_cols.append(item)
                train_vals.append(1)
                train_prev_vals.append(item_prev)
                train_times.append(item_time[1])
                train_prev_times.append(item_prev_time[1])

        self.sp_train = sp.coo_matrix((train_vals, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_prev = sp.coo_matrix((train_prev_vals, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_times = sp.coo_matrix((train_times, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_train_prev_times = sp.coo_matrix((train_prev_times, (train_rows, train_cols)),
                shape=(self.num_users, self.num_items))

        # Sparse validation matrices
        val_rows = []
        val_cols = []
        val_vals = []
        val_prev_vals = []
        val_times = []
        val_prev_times = []
        for user in self.val_set:
            item = self.val_set[user][0]
            item_prev = self.val_set[user][1]
            item_time = self.val_times[user][0]
            item_prev_time = self.val_times[user][1]
            if item == -1 or item_prev == -1:
                continue

            val_rows.append(user)
            val_cols.append(item)
            val_vals.append(1)
            val_prev_vals.append(item_prev)
            val_times.append(item_time)
            val_prev_times.append(item_prev_time)

        self.sp_val = sp.coo_matrix((val_vals, (val_rows, val_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_val_prev = sp.coo_matrix((val_prev_vals, (val_rows, val_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_val_times = sp.coo_matrix((val_times, (val_rows, val_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_val_prev_times = sp.coo_matrix((val_prev_times, (val_rows, val_cols)),
                shape=(self.num_users, self.num_items))

        # Sparse test matrices
        test_rows = []
        test_cols = []
        test_vals = []
        test_prev_vals = []
        test_times = []
        test_prev_times = []
        for user in self.test_set:
            item = self.test_set[user][0]
            item_prev = self.test_set[user][1]
            item_time = self.test_times[user][0]
            item_prev_time = self.test_times[user][1]
            if item == -1 or item_prev == -1:
                continue

            test_rows.append(user)
            test_cols.append(item)
            test_vals.append(1)
            test_prev_vals.append(item_prev)
            test_times.append(item_time)
            test_prev_times.append(item_prev_time)

        self.sp_test = sp.coo_matrix((test_vals, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_prev = sp.coo_matrix((test_prev_vals, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_times = sp.coo_matrix((test_times, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))
        self.sp_test_prev_times = sp.coo_matrix((test_prev_times, (test_rows, test_cols)),
                shape=(self.num_users, self.num_items))

        self.val_prev_cats = None
        self.test_prev_cats = None
            
    def generate_train_batch_sp(self):
        # Subtract 1 to account for missing 0 index
        user_indices = self.sp_train.row - 1
        prev_indices = self.sp_train_prev.data - 1
        pos_indices = self.sp_train.col - 1
        neg_indices = np.random.randint(1, self.sp_train.shape[1],
                size=len(self.sp_train.row), dtype=np.int32) - 1

        # Convert from indices to one hot matrices
        users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]
        neg_items = self.item_one_hot[neg_indices]

        # Horizontally stack sparse matrices to get single positive
        # and negative feature matrices
        pos_feats = sp.hstack([users, prev_items, pos_items])
        neg_feats = sp.hstack([users, prev_items, neg_items])

        if self.args.features == 'categories':
            # Join with categories
            train_prev_cats = self.cat_mat[prev_indices]
            train_pos_cats = self.cat_mat[pos_indices]
            train_neg_cats = self.cat_mat[neg_indices]
            pos_feats = sp.hstack([pos_feats, train_prev_cats, train_pos_cats])
            neg_feats = sp.hstack([neg_feats, train_prev_cats, train_neg_cats])

        elif self.args.features == 'time':
            # Join with temporal data
            prev_times = self.sp_train_prev_times.data
            next_times = self.sp_train_times.data
            pos_feats = sp.hstack([pos_feats, prev_times[:, None], next_times[:, None]])
            neg_feats = sp.hstack([neg_feats, prev_times[:, None], next_times[:, None]])

        elif self.args.features == 'content':
            # Join with content data
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            neg_item_content = self.item_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])

        elif self.args.features == 'geo':
            # Join with geographical data
            pos_geo = self.geo_feats[pos_indices]
            neg_geo = self.geo_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, pos_geo])
            neg_feats = sp.hstack([neg_feats, neg_geo])

        return (users, pos_feats, neg_feats)

    def generate_val_batch_sp(self, items_per_user=10):
        user_indices = np.repeat(self.sp_val.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_val_prev.data, items_per_user) - 1
        pos_indices = np.repeat(self.sp_val.col, items_per_user) - 1
        neg_indices = np.random.randint(1, self.sp_val.shape[1],
                size=len(self.sp_val.row)*items_per_user, dtype=np.int32) - 1

        # Convert from indices to one hot matrices
        users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]
        neg_items = self.item_one_hot[neg_indices]

        # Horizontally stack sparse matrices to get single positive
        # and negative feature matrices
        pos_feats = sp.hstack([users, prev_items, pos_items])
        neg_feats = sp.hstack([users, prev_items, neg_items])

        if self.args.features == 'categories':
            # Join with categories
            if self.val_prev_cats is None:
                self.val_prev_cats = self.cat_mat[prev_indices]
                self.val_pos_cats = self.cat_mat[pos_indices]
            self.val_neg_cats = self.cat_mat[neg_indices]
            pos_feats = sp.hstack([pos_feats, self.val_prev_cats, self.val_pos_cats])
            neg_feats = sp.hstack([neg_feats, self.val_prev_cats, self.val_neg_cats])

        elif self.args.features == 'time':
            # Join with temporal data
            prev_times = np.repeat(self.sp_val_prev_times.data, items_per_user)[:, None]
            next_times = np.repeat(self.sp_val_times.data, items_per_user)[:, None]
            pos_feats = sp.hstack([pos_feats, prev_times, next_times])
            neg_feats = sp.hstack([neg_feats, prev_times, next_times])

        elif self.args.features == 'content':
            # Join with content data
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            neg_item_content = self.item_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])

        elif self.args.features == 'geo':
            # Join with geographical data
            pos_geo = self.geo_feats[pos_indices]
            neg_geo = self.geo_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, pos_geo])
            neg_feats = sp.hstack([neg_feats, neg_geo])

        return (users, pos_feats, neg_feats)

    def generate_test_batch_sp(self, items_per_user=10):
        user_indices = np.repeat(self.sp_test.row, items_per_user) - 1
        prev_indices = np.repeat(self.sp_test_prev.data, items_per_user) - 1
        pos_indices = np.repeat(self.sp_test.col, items_per_user) - 1
        neg_indices = np.random.randint(1, self.sp_test.shape[1],
                size=len(self.sp_test.row)*items_per_user, dtype=np.int32) - 1

        # Convert from indices to one-hot matrices
        users = self.user_one_hot[user_indices]
        prev_items = self.item_one_hot[prev_indices]
        pos_items = self.item_one_hot[pos_indices]
        neg_items = self.item_one_hot[neg_indices]

        # Horizontally stack sparse matrices to get single positive
        # and negative feature matrices
        pos_feats = sp.hstack([users, prev_items, pos_items])
        neg_feats = sp.hstack([users, prev_items, neg_items])

        if self.args.features == 'categories':
            # Join with categories
            if self.test_prev_cats is None:
                self.test_prev_cats = self.cat_mat[prev_indices]
                self.test_pos_cats = self.cat_mat[pos_indices]
            self.test_neg_cats = self.cat_mat[neg_indices]
            pos_feats = sp.hstack([pos_feats, self.test_prev_cats, self.test_pos_cats])
            neg_feats = sp.hstack([neg_feats, self.test_prev_cats, self.test_neg_cats])

        elif self.args.features == 'time':
            # Join with temporal data
            prev_times = np.repeat(self.sp_test_prev_times.data, items_per_user)[:, None]
            next_times = np.repeat(self.sp_test_times.data, items_per_user)[:, None]
            pos_feats = sp.hstack([pos_feats, prev_times, next_times])
            neg_feats = sp.hstack([neg_feats, prev_times, next_times])

        elif self.args.features == 'content':
            # Join with content data
            user_content = self.user_feats[user_indices]
            pos_item_content = self.item_feats[pos_indices]
            neg_item_content = self.item_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, user_content, pos_item_content])
            neg_feats = sp.hstack([neg_feats, user_content, neg_item_content])

        elif self.args.features == 'geo':
            # Join with geographical data
            pos_geo = self.geo_feats[pos_indices]
            neg_geo = self.geo_feats[neg_indices]
            pos_feats = sp.hstack([pos_feats, pos_geo])
            neg_feats = sp.hstack([neg_feats, neg_geo])

        return (users, pos_feats, neg_feats)

