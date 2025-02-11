from random import sample
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
import scipy.sparse as sp
from pynndescent import NNDescent
from collections import defaultdict
from sklearn import metrics


def hierarchical_clustering(x,initial_rank=None, distance='euclidean', verbose=False, ann_threshold=40000,layers=5):
    ### Performing finch clustering
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    x = x.astype(np.float32)
    min_sim = None

    # calculate pairwise similarity orig_dis to find the nearest neighbor and obtain the adj matrix
    adj, orig_dist, first_neighbors, _ = clust_rank(    ###x:feature:(data,aug1,agu2) in latent space
        x,
        initial_rank,
        distance,
        verbose=verbose,
        ann_threshold=ann_threshold
    )

    initial_rank = None

    u, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], u, x) # obtain the centroids according to the partition and raw data
    cluster = defaultdict(list)
    outliers_dist = defaultdict(list)

    for i in range(0, len(u)):  # u: current partition, c: all partitions
        cluster[u[i]].append(i)
        outliers_dist[u[i]].append(i)

    lowest_level_centroids = mat

    ''' save centroids of the bottom layer (layer 0)'''
    lowest_centroids = torch.Tensor(lowest_level_centroids).cuda()
    results['centroids'].append(lowest_centroids)

    if verbose:
        print('Level/Partition 0: {} clusters'.format(num_clust))

    c_ = c  # transfer value first and then mask
    num_clust = [num_clust] #int->list
    partition_clustering = []
    adj, orig_dist, first_neighbors, knn_index = clust_rank(
        mat,
        initial_rank,
        distance,
        verbose=verbose,
        ann_threshold=ann_threshold
    )

    u, num_clust_curr = get_clust(adj, orig_dist, min_sim)  #u = group
    partition_clustering.append(u)  # all partitions (u: current partition)

    c_, mat = get_merge(c_, u, x)    ##mat:center of clustering
    c = np.column_stack((c, c_))     #### Cumulatively, in each layer, which category each sample belongs to
    num_clust.append(num_clust_curr)

    """ save multiple partitions """
    # save 131 32 7 from [533, 131, 32, 7, 2]
    for i in range(0, len(c[0])):
        im2cluster = [int(n[i]) for n in c]
        im2cluster = torch.LongTensor(im2cluster).cuda()
        results['im2cluster'].append(im2cluster)

    return c, num_clust, partition_clustering, lowest_level_centroids, results

def clust_rank(
        mat,
        initial_rank=None,
        metric='cosine',
        verbose=False,
        ann_threshold=40000):
    knn_index = None
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ann_threshold:
        # If the sample size is smaller than threshold, use metric to calculate similarity.
        # If the sample size is larger than threshold, use PyNNDecent to speed up the calculation of nearest neighbor
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=metric)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if verbose:
            print('Using PyNNDescent to compute 1st-neighbours at this step ...')
        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=metric,
            verbose=verbose)
        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        if verbose:
            print('Step PyNNDescent done ...')

    sparce_adjacency_matrix = sp.csr_matrix(
        (np.ones_like(initial_rank, dtype=np.float32),
         (np.arange(0, s), initial_rank)),
        shape=(s, s))  # join adjacency matrix based on Initial rank

    return sparce_adjacency_matrix, orig_dist, initial_rank, knn_index

def cool_mean(data, partition, max_dis_list=None):
    s = data.shape[0]
    un, nf = np.unique(partition, return_counts=True)

    row = np.arange(0, s)
    col = partition
    d = np.ones(s, dtype='float32')

    if max_dis_list is not None:
        for i in max_dis_list:
            data[i] = 0
        nf = nf - 1

    umat = sp.csr_matrix((d, (row, col)), shape=(s, len(un)))
    cluster_rep = umat.T @ data
    cluster_mean_rep = cluster_rep / nf[..., np.newaxis]

    return cluster_mean_rep

def get_clust(a, orig_dist, min_sim=None):
    # connect nodes based on adj, orig_dist, min_sim
    # build the graph and obtain multiple components/clusters
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust

def get_merge(partition, group, data):
    # get_merge([], group, x)
    # u/group: (n,)  data/x: (n, dim)
    if len(partition) != 0:
        _, ig = np.unique(partition, return_inverse=True)
        partition = group[ig]
    else:
        partition = group

    mat = cool_mean(data, partition, max_dis_list=None) # mat: computed centroids(k,dim)
    # data: (n, dim)   partition: (n,)  return:(k, dim)
    return partition, mat

def Contrastive_Loss(data,posi_num,neg_num,cluster_result,T):
    p0_label = {}  # dict (key:index, value:label)
    label_index = {}  # dict
    index_u = {}
    index=np.arange(cluster_result['im2cluster'][0].shape[0])
    for u in range(0, index.shape[0]):
        index_u[index[u].item()] = u
        p0_label[index[u].item()] = cluster_result['im2cluster'][0][index[u]].item()
    # find keys(ids) with same value(cluster label) in dict p0_label
    for key, value in p0_label.items():
        label_index.setdefault(value, []).append(key)

    posid = {}
    negid = {}
    neg_instances = [[] for _ in range(len(p0_label))]
    pos_instances = [[] for _ in range(len(p0_label))]

    for i in p0_label:
        posid[i] = label_index[p0_label[i]].copy()  # all candidate pos instances(if not enough, copy itself)
        if (len(posid[i])) < posi_num:
            for _ in range(0, posi_num - len(posid[i])):
                posid[i].append(i)
        negid[i] = [x for x in index.tolist() if x not in posid[i]]

        if (len(posid[i])) >= posi_num:
            posid[i] = sample(posid[i], posi_num)  # if len = self.posi, preserve
        negid[i] = sample(negid[i], neg_num)

        for m in range(len(posid[i])):
            pos_instances[index_u[i]].append(data[index_u[posid[i][m]]])  # all candidate pos instances(if not enough, copy itself)
        pos_instances[index_u[i]] = torch.stack(pos_instances[index_u[i]])

        for n in range(len(negid[i])):
            neg_instances[index_u[i]].append(data[index_u[negid[i][n]]])
        neg_instances[index_u[i]] = torch.stack(neg_instances[index_u[i]])
    pos_instances=torch.stack(pos_instances)
    neg_instances=torch.stack(neg_instances)

    ##compute contrastive loss
    pos_instances=torch.reshape(pos_instances,(pos_instances.shape[0],pos_instances.shape[2],pos_instances.shape[1]))
    neg_instances = torch.reshape(neg_instances,
                                  (neg_instances.shape[0], neg_instances.shape[2], neg_instances.shape[1]))
    data = data.unsqueeze(1)
    data_abs=data.norm(dim=2)
    pos_abs=pos_instances.norm(dim=1)
    neg_abs = neg_instances.norm(dim=1)
    pos_sim_matrix = torch.einsum('nab,nbc->nac', data, pos_instances) / torch.einsum('na,nc->nac', data_abs, pos_abs)
    neg_sim_matrix = torch.einsum('nab,nbc->nac', data, neg_instances) / torch.einsum('na,nc->nac', data_abs, neg_abs)
    pos_matrix = torch.exp(pos_sim_matrix / T)
    neg_matrix = torch.exp(neg_sim_matrix / T)
    pos_matrix=pos_matrix.squeeze(dim=1)
    neg_matrix=neg_matrix.squeeze(dim=1)
    loss = pos_matrix.sum(dim=1) / neg_matrix.sum(dim=1)
    loss = -torch.log(loss).mean()
    return loss   ##(batch,number,dim)



if __name__=='__main__':
    x=np.random.randn(128,64)
    c, num_clust, partition_clustering, lowest_level_centroids, cluster_result = hierarchical_clustering(
        x, initial_rank=None, distance='euclidean', verbose=False, ann_threshold=40000,layers=2)
    print(len(cluster_result['im2cluster']))
    print('111',cluster_result['im2cluster'][0])
    x=torch.tensor(x)
    loss=Contrastive_Loss(x,posi_num=3,neg_num=3,cluster_result=cluster_result,T=0.5)
    print(loss)

