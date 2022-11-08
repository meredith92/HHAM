#
# import scipy.sparse as sp
import numpy as np
#
# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape
#
# def preprocess_graph(adj):
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return sparse_to_tuple(adj_normalized)
#
# A = sp.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# # B=A.diagonal()[np.newaxis, :]
# # C=sp.dia_matrix((B, [0]),shape=A.shape)
# D=preprocess_graph(A)
# adj_orig = A

#
# B=sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#
# C = adj_orig - B
#
# D=C.eliminate_zeros()
#
# adj_triu = sp.triu(C)
#
# adj_tuple = sparse_to_tuple(adj_triu)
x1 = np.arange(6.0).reshape((2, 3))
x2 = np.arange(6.0).reshape((2,3))
D=np.dot(x1, x2)
print(x1)
print(x2)
print(D)