import scipy.sparse as sp
import networkx as nx
import numpy as np

filename= 'output/am/am_e2v_embeddings.txt'

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
    feature=sp.csr_matrix(allx).tolil()

filename1='am/am.txt'
with open(filename1, 'r') as f:
    # edgenum=len(f.readlines())
    edgelist=[]
    for line in f:
        items=line.strip().split()
        edgelist.append((nodelist.index(items[0]),nodelist.index(items[1])))

    graph = nx.DiGraph(edgelist)
    adj = nx.adjacency_matrix(graph)





