# coding=utf-8
import numpy as np
import scipy.sparse as sp
from normalization import fetch_normalization
from scipy.sparse import coo_matrix

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class Sampler:
    """Sampling the input graph data."""
    def __init__(self, adj):
        adj = coo_matrix(adj)
        self.adj = adj
        self.train_adj = sparse_to_tuple(adj)

    def randomedge_sampler(self, percent):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        if percent >= 1.0:
            return self.adj

        nnz = len(self.train_adj[1])
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        test = self.train_adj[0][perm, 0]
        r_adj = sp.coo_matrix((self.train_adj[1][perm],
                               (self.train_adj[0][perm, 0],
                                self.train_adj[0][perm, 1])),
                              shape=self.train_adj[2])

        return r_adj.todense()

    def randomvertex_sampler(self, percent):
        """
        Randomly drop vertexes.
        """
        if percent >= 1.0:
            return self.adj
        # print ("start")
        row = np.unique(self.train_adj[0][:, 0])
        data = self.train_adj[1]
        n_row = len(row)
        perm = np.random.permutation(n_row)
        preserve_row = int(n_row*percent)
        perm = perm[:preserve_row]
        preserve_row_set = set(row[perm])
        new_row = list()
        new_col = list()
        new_data = list()
        for i in range(len(self.train_adj[1])):
            # print (i)
            tmp_row = self.train_adj[0][i, 0]
            tmp_col = self.train_adj[0][i, 1]
            if tmp_row in preserve_row_set and tmp_col in preserve_row_set:
                new_row.append(tmp_row)
                new_col.append(tmp_col)
                new_data.append(self.train_adj[1][i])
        r_adj = sp.coo_matrix((new_data,
                               (new_row,
                                new_col)),
                              shape=self.train_adj[2])
        return r_adj.todense()


    def degree_sampler(self, percent, normalization):
        """
        Randomly drop edge wrt degree (high degree, low probility).
        """
        print('not implemented yet')
        exit();
