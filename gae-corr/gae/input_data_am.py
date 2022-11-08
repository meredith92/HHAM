import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def load_data():
    filename= 'am/am_e2v_embeddings02.txt'
    # filename= 'douban/douban_e2v_embeddings_new.txt'

    print('loading data...')

    nodelist=[]

    with open(filename, 'r') as f:
        nodenum, embdim=f.readline().strip().split()
        i=0
        allx = np.zeros((int(nodenum), int(embdim)))
        for line in f:
            items = line.strip().split()
            nodelist.append(items[0])
            allx[i]=items[1:]
            i=i+1
        features=sp.csr_matrix(allx).tolil()

        emb_event = features.toarray()
        emb_event=np.around(emb_event,decimals=2)
        # num=len(emb_event)
        cosinesim=cosine_similarity(emb_event)


    filename1='am/am.txt'
    # filename1='douban/douban.txt'

    with open(filename1, 'r') as f:
        # edgenum=len(f.readlines())
        edgelist=[]
        for line in f:
            items=line.strip().split()
            edgelist.append((nodelist.index(items[0]),nodelist.index(items[1])))

        graph = nx.DiGraph(edgelist)
        adj = nx.adjacency_matrix(graph)
    return adj, features ,cosinesim