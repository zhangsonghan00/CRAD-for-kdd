import torch
import torch.nn as nn
import math
import copy
import random
from cluster import hierarchical_clustering,Contrastive_Loss

class CCPerceptron(nn.Module):
    def __init__(self, seq_len, n_feature, n_layer):
        super(CCPerceptron, self).__init__()
        self.seq_len = seq_len
        self.n_feature = n_feature
        self.use_contrastive=False

        self.Time_Linear = nn.ModuleList()
        self.Feature_Linear = nn.ModuleList()
        self.Time_Linear.append(nn.Linear(seq_len, seq_len))
        self.Feature_Linear.append(nn.Linear(n_feature,n_feature))
        for _ in range(n_layer-1):
            self.Time_Linear.append(nn.Sequential(
                nn.ReLU(True),
                nn.Linear(seq_len, seq_len)
            ))
            self.Feature_Linear.append(nn.Sequential(
                nn.ReLU(True),
                nn.Linear(n_feature, n_feature)
            ))

        self.fusion = nn.Linear(2 * n_feature, n_feature)
        self.sigmoid = nn.Sigmoid()

    def augment(self,input_feat,drop_dim=1,random_mask=True,mask_percent=0.1):
        aug_input_feat=copy.deepcopy(input_feat)
        total_num=aug_input_feat.shape[drop_dim]
        drop_percent=mask_percent
        drop_feat_num=int(total_num*drop_percent)
        if random_mask:
            drop_idx=random.sample([i for i in range(total_num)],drop_feat_num)
            aug_input_feat[:,drop_idx,:]=0
        else:
            ind=total_num-drop_feat_num
            aug_input_feat[:,ind:,:]=0
        return aug_input_feat

    def forward(self, x,training=False):
        # x = [batch_size, seq_len, n_feature]

        if training:
            time_enc = torch.transpose(x, -1, -2)
            for layer in self.Time_Linear:
                time_enc = layer(time_enc)
            time_enc = torch.transpose(time_enc, -1, -2)

            ### each layer in Feature_Linear
            fea_enc = x
            for layer in self.Feature_Linear:
                fea_enc = layer(fea_enc)
            if self.use_contrastive:
                x_aug=self.augment(x)
                time_enc_aug = torch.transpose(x_aug, -1, -2)
                for layer in self.Time_Linear:
                    time_enc_aug = layer(time_enc_aug)
                time_enc_aug = torch.transpose(time_enc_aug, -1, -2)

                ### each layer in Feature_Linear
                fea_enc_aug = x_aug
                for layer in self.Feature_Linear:
                    fea_enc_aug = layer(fea_enc_aug)
        else:
            time_enc = torch.transpose(x, -1, -2)
            for layer in self.Time_Linear:
                time_enc = layer(time_enc)
            time_enc = torch.transpose(time_enc, -1, -2)

            ### each layer in Feature_Linear
            fea_enc=x
            for layer in self.Feature_Linear:
                fea_enc = layer(fea_enc)

        if training and self.use_contrastive:
            x1=torch.cat([time_enc,time_enc_aug],dim=0).reshape(x.shape[0]*2,-1)
            x1=x1.cpu().detach().numpy()
            c, num_clust, partition_clustering, lowest_level_centroids, cluster_result = hierarchical_clustering(
                x1, initial_rank=None, distance='euclidean', verbose=False, ann_threshold=40000, layers=2)
            x1=torch.tensor(x1).to(time_enc.device)
            contrastive_loss1=Contrastive_Loss(x1, posi_num=3, neg_num=3,cluster_result=cluster_result,T=0.5)

            x2 = torch.cat([fea_enc, fea_enc_aug], dim=0).reshape(x.shape[0] * 2, -1)
            x2 = x2.cpu().detach().numpy()
            c, num_clust, partition_clustering, lowest_level_centroids, cluster_result = hierarchical_clustering(
                x2, initial_rank=None, distance='euclidean', verbose=False, ann_threshold=40000, layers=2)
            x2 = torch.tensor(x2).to(fea_enc.device)
            contrastive_loss2 = Contrastive_Loss(x2, posi_num=3, neg_num=3, cluster_result=cluster_result, T=0.5)
            contrastive_loss=(contrastive_loss1+contrastive_loss2)/2
        else:
            contrastive_loss=torch.zeros([1])[0].to(x.device)

        ##concatenate
        enc = torch.cat([fea_enc, time_enc], dim=-1)

        ####decoder
        enc = self.fusion(enc)
        out = self.sigmoid(enc)

        return out,contrastive_loss
