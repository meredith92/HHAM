from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import keras

import multiprocessing as mp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data_am import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpu_available=tf.test.is_gpu_available()
# print(gpu_available)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Settings

config=tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6,use_per_session_threads=True)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset


# Load data
adj, features, cosinesimi = load_data()
# emb_event = features.toarray()
# emb_event =1 / (1 + np.exp(-emb_event))
# num=len(emb_event)
# adj=np.zeros((num,num))
# for i in range(0, num-1):
#     for j in range(1,num):
#         s = np.dot(emb_event[i],emb_event[j])/(np.linalg.norm(emb_event[i])*np.linalg.norm(emb_event[i]))
#         adj[i][j]=s
#         adj[j][i]=s


# adj_weight=1 / (1 + np.exp(-adj))
# np.where(adj_weight>0.5,adj_weight,1)
# np.where(adj_weight<=0.5, adj_weight,0)
print('feature',sum(features.toarray()[0]))


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# print('split data')
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
# print('preprocessing')
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

        # emb = emb*emb_event

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)*cosinesimi
    adj_rec = np.dot(emb, emb.T)

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score,preds,preds_neg


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # print('start')
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    # print('feedupdate')
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    # print('run')

    # coord = tf.train.Coordinator()
    # threads=tf.train.start_queue_runners(sess,coord=coord)
    # try:
    #     step=0
    #     while not coord.should_stop():
    #
    outs= sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    #     except tf.errors.OutOfRangeError:
    #     print('Done training for %d epoches,%d steps' % (FLAGS.num_epoches, step))
    # finally:
    #     coord.request_stop()
    # coord.join(threads)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr, _, _= get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

test_edges=[[140,114],[140,115],[140,116],[140,117],[140,118],[140,119],[140,120],[140,121],[140,122],[140,123],[140,124],[140,125],
            [140,126],[140,127],[140,128],[140,129],[140,130],[140,131],[140,132],[140,133],[140,134],[140,135],[140,136],
            [140,137],[140,138],[140,139],[140,141],[140,142],[140,143],[140,144],[140,145],[140,146],[140,147],[140,148],
            [140,149],[140,150],[140,151],[140,152],[140,153],[140,154],[140,155],[140,156],[140,157],[140,158],[140,159],
            [140,160],[140,161],[140,162]]

test_edges_false=[[141,114],[141,115],[141,116],[141,117],[141,118],[141,119],[141,120],[141,121],[141,122],[141,123],[141,124],[141,125],
            [141,126],[141,127],[141,128],[141,129],[141,130],[141,131],[141,132],[141,133],[141,134],[141,135],[141,136],
            [141,137],[141,138],[141,139],[141,140],[141,142],[141,143],[141,144],[141,145],[141,146],[141,147],[141,148],
            [141,149],[141,150],[141,151],[141,152],[141,153],[141,154],[141,155],[141,156],[141,157],[141,158],[141,159],
            [141,160],[141,161],[141,162]]

roc_score, ap_score,preds,preds_neg = get_roc_score(test_edges, test_edges_false)
# print (test_edges)
# print (test_edges_false)
print(preds)
print(preds_neg)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
