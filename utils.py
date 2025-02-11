import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn import preprocessing
from torch.utils.data import Dataset

def Normalization(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    # x = x.values
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    return x,min_max_scaler

def plot_result(result):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    sns.displot(result)
    # plt.xlim([0,0.2])
    # plt.ylim([0,1600])
    plt.show()


def train_model(train_loader, val_loader, model, optimizer, loss_fn, scaler, device,num_epochs,scheduler):
    train_loss_all=[]
    val_loss_all=[]
    for epoch in range(num_epochs):
        print('Epoch:', epoch + 1)
        #model.train()
        model.train()
        loop = tqdm(train_loader)
        train_loss=[]
        for batch_idx, (x) in enumerate(loop):
            x = x.to(device)
            # forward
            with torch.cuda.amp.autocast():
                pred,contrastive_loss = model(x,training=True)
                mseloss=loss_fn(pred,x)
                loss=mseloss+contrastive_loss / (contrastive_loss / mseloss).detach()
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            train_loss.append(loss.item())
        scheduler.step()
        # model.eval()
        val_loss = []
        model.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                pred,_= model(x,training=False)
                mseloss = loss_fn(pred,x)
                loss = mseloss
                val_loss.append(loss.item())
        train_loss=np.mean(train_loss)
        val_loss=np.mean(val_loss)
        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        print('train loss:',train_loss)
        print('validation loss:',val_loss)

    # save model
    print('------>Saving checkpoint')
    torch.save(model.state_dict(), 'MLP.pth')
    return model.eval(),train_loss_all,val_loss_all

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ts=torch.FloatTensor(self.data[index])
        return ts

def test(test_loader,device,loss_fn,model,model_name):
    test_loss=[]
    mean_dim=[1,2]
    mean_dim=-1  ##for point-wise AD
    with torch.no_grad():
        for i,x in enumerate(test_loader):
            x = x.to(device)
            if model_name=='VAE':
                pred,mu,sigma=model(x,training=False)
            else:
                pred,_ = model(x,training=False)
            loss = torch.mean(loss_fn(pred,x), mean_dim).reshape(-1).tolist()   ##for mlp_AE(uni)
            pred1=pred.detach().cpu().numpy()
            test_loss+=loss
            if i==0:
                y_pred=pred1
            else:
                y_pred=np.concatenate((y_pred,pred1),axis=0)
    return test_loss,y_pred

def check_accuracy(label,pred):
    pred=np.array(pred)
    label=np.array(label)
    TP=(pred*label).sum()
    FP=pred.sum()-(pred*label).sum()
    FN=label.sum()-(pred*label).sum()
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1_score=2*recall*precision/(recall+precision)
    return precision,recall,f1_score

def get_bestF1(lab, scores, PA=False):
    scores = np.array(scores)
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones

    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    new_scores = scores[sortid]

    if PA:
        lab_diff = np.insert(lab, len(lab), 0) - np.insert(lab, 0, 0)
        a_st = np.arange(len(lab) + 1)[lab_diff == 1]
        a_ed = np.arange(len(lab) + 1)[lab_diff == -1]

        thres_a = np.array([np.max(scores[a_st[i]:a_ed[i]]) for i in range(len(a_st))])
        sort_a_id = np.flip(np.argsort(thres_a))  # big to small
        cum_a = np.cumsum(a_ed[sort_a_id] - a_st[sort_a_id])

        last_thres = np.inf
        TPs = np.zeros_like(new_lab)
        for i, a_id in enumerate(sort_a_id):
            TPs[(thres_a[a_id] <= new_scores) & (new_scores < last_thres)] = cum_a[i - 1] if i > 0 else 0
            last_thres = thres_a[a_id]
        TPs[new_scores < last_thres] = cum_a[-1]
    else:
        TPs = np.cumsum(-new_lab) + ones

    FPs = np.cumsum(new_lab - 1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs

    N = len(lab) - np.flip(TPs > 0).argmax()  ###anyway:找出label中的1的个数
    TPRs = TPs[:N] / ones  ###让TPR很大
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)


    anomaly_ratio = ones / len(lab)
    FPR_bestF1_TPR1 = anomaly_ratio / (1 - anomaly_ratio) * (2 / F1s[maxid] - 2)
    TPR_bestF1_FPR0 = F1s[maxid] / (2 - F1s[maxid])
    return {'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid],
            'FPR': FPRs[maxid], 'maxid': maxid,
            'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = np.random.randn(495000)
    plt.figure()
    sns.displot(data)
    plt.show()



