import tensorflow as tf
import numpy as np
import argparse
from eventnet import *
import pickle
import time
import os
import networkx as nx
from sklearn import preprocessing


import event2vec


def main():
    ## Parameters setting
    #           beta    rep_dim     epochs      batch_size      learning_rate
    # Movielens: 30        64        2000           128             0.01
    # DBLP     : 30        64        200            128             0.01
    # Douban   : 30        64        200            128             0.01
    # IMDB     : 2         64        200            128             0.01
    # Yelp     : 30        64        2000           128             0.01
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', default=30)
    parser.add_argument('--representation_dim', default=64)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--learning_rate', default=0.01)

    # parser.add_argument('--graph_file', default='dblp/dblp.txt')

    # parser.add_argument('--graph_file', default='douban/douban.txt')
    # parser.add_argument('--graph_file', default='imdb/imdb.txt')
    # parser.add_argument('--graph_file', default='yelp/yelp.txt')
    parser.add_argument('--graph_file', default='am/am.txt')

    parser.add_argument('--event_id_idx', default=[0]) # for dblp | douban | yelp | am
    # parser.add_argument('--event_id_idx', default=[1]) # for imdb
     
    # parser.add_argument('--node_types', default=['a', 'p', 'c', 't']) # for dblp
    # parser.add_argument('--node_types', default=['m', 'a', 'd', 'u']) # douban
    # parser.add_argument('--node_types', default=['a', 'm', 'u', 'd']) # imdb
    # parser.add_argument('--node_types', default=['b', 'u', 'l', 'c']) # yelp
    parser.add_argument('--node_types', default=['e', 'p', 's', 'd','m'])  # for am

    # parser.add_argument('--output_file', default='output/dblp/dblp_e2v_embeddings.txt')
    # parser.add_argument('--output_file', default='output/douban/douban_e2v_embeddings.txt')
    # parser.add_argument('--output_file', default='output/imdb/imdb_e2v_embeddings.txt')
    # parser.add_argument('--output_file', default='output/yelp/yelp_e2v_embeddings.txt')
    parser.add_argument('--output_file', default='output/am/am_e2v_embeddings.txt')

    args = parser.parse_args()
    train(args)

def intraedge(event_dict,node_types):
    eventnum=len(event_dict)
    edgeweight =np.zeros((eventnum,eventnum))
    intraedge = np.zeros((eventnum, eventnum))
    for i in range(eventnum-1):
        sumnode=0
        for j in range(i+1,eventnum):
            samenode = 0
            for nodetype in node_types:
                if (nodetype in event_dict[i]) &(nodetype in event_dict[j]):
                    list1=event_dict[i][nodetype]
                    list2=event_dict[j][nodetype]
                    set1=set(list1)
                    set2=set(list2)
                    samenode= samenode+len(set1&set2)
            edgeweight[i][j]= samenode
            edgeweight[j][i]= samenode
            sumnode+=samenode
                # if samenode>2:
        if sumnode !=0:
            intraedge[i]=edgeweight[i]/sumnode
    intraedge =intraedge+intraedge.T
    return intraedge,edgeweight

# def make_adj(args):
#     with open(args.output_file) as f:
#         nodelist=[]
#         for line in f:
#             items = line.strip().split()
#             nodelist.append(items[0])
#
#     with open(args.graph_file, 'r') as f:
#         edgelist = []
#         for line in f:
#             items = line.strip().split()
#             edgelist.append((nodelist.index(items[0]), nodelist.index(items[1])))
#     return edgelist

def train(args):
    data = LoadData()
    data.load_data(filename=args.graph_file, event_id_idx=args.event_id_idx)
    
    nodes_num = data.nodes_num
    event_num = data.event_num
    node_ind = data.node_ind
    event_dict_int = data.event_dict_int
    
    node_types = args.node_types     
    
    del data
    
    event = EventNet(event_dict=event_dict_int, 
                    nodes_num=nodes_num,
                    event_num=event_num,
                    node_types=node_types)
    
    node_deg = event.node_deg
    inc_mat = event.inc_mat
    edgedata,edgeweight =intraedge(event_dict_int,node_types)
    inc_mat['edge'] = edgedata

    print('number of events: {}'.format(event_num))
    print(nodes_num)
    model = event2vec.EVENT2VEC(nodes_num=nodes_num, 
                                node_types=node_types,
                                nodes_ind=node_ind,
                                event_dict=event_dict_int,
                                beta=args.beta,
                                rep_size=args.representation_dim,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate
                                )
    
    
    for node_type in node_types:
        inc_mat[node_type] = inc_mat[node_type].T

    
    t1 = time.time()
    model.train(inc_mat)
    t2 = time.time()
    print('training time: %s'%(t2-t1))

    model.save_embeddings(args.output_file, inc_mat, node_deg)
    
    

if __name__ == '__main__':
    main()
    