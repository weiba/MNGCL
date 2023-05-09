import pickle
from datetime import datetime
from sklearn import metrics
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import yaml
from yaml import SafeLoader
from time import perf_counter as t
import numpy as np
from mngcl import MNGCL
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn import linear_model
import warnings
from gcn import GCN
from torch_geometric.utils import dropout_adj
warnings.filterwarnings("ignore")

cross_val=10

def load_label_single(path):
    label = np.loadtxt(path + "label_file-P-"+cancerType+".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-"+cancerType+".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg

def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    # pos_label：Positive sample index
    # neg_label：Negative sample index
    # l：number of genes
    # l1：Number of positive samples
    # l2：number of negative samples
    # i：number of folds
    pos_test = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_test))
    neg_test = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_test))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_test)):
        indexs2[pos_test[j]] = True
    for j in range(len(neg_test)):
        indexs2[neg_test[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    te_mask = torch.from_numpy(np.array(indexs2))
    return tr_mask, te_mask
   
def train(mask):
    model.train()
    optimizer.zero_grad()
    x = data.x
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    #feature mask
    x_1 = F.dropout(x, drop_feature_rate_1)
    x_2 = F.dropout(x, drop_feature_rate_2)
    x_3 = F.dropout(x, drop_feature_rate_3)
    #edge dropout
    ppiAdj_index = dropout_adj(ppiAdj_index, p=drop_edge_rate_1, force_undirected=True)[0]
    pathAdj_index = dropout_adj(pathAdj_index, p=drop_edge_rate_2, force_undirected=True)[0]
    goAdj_index = dropout_adj(goAdj_index, p=drop_edge_rate_3, force_undirected=True)[0]
    pred1,pred2,pred3,_,conloss= model(ppiAdj_index,pathAdj_index,goAdj_index,x_1,x_2,x_3)
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss2 = F.binary_cross_entropy_with_logits(pred2[mask], Y[mask])
    loss3 = F.binary_cross_entropy_with_logits(pred3[mask], Y[mask])
    loss = 0.55*conloss+0.15*loss1+0.15*loss2+0.15*loss3
    loss.backward()
    optimizer.step()
    return loss.item()

def LogReg(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:,1]
    return pre

@torch.no_grad()
def test(mask1, mask2):
    model.eval()
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    _,_,_,emb,_ = model(ppiAdj_index,pathAdj_index,goAdj_index,data.x,data.x,data.x)
    train_x = torch.sigmoid(emb[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(emb[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy()
    pred = LogReg(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPDB')
    parser.add_argument('--cancer_type', type=str, default='pan-cancer')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    dataPath = "data/"+args.dataset+"/"
    cancerType = args.cancer_type
    seed = config['seed']
    LR = config['LR']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_edge_rate_3 = config['drop_edge_rate_3']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_feature_rate_3 = config['drop_feature_rate_3']
    tau = config['tau']
    EPOCH = config['EPOCH']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    data = torch.load(dataPath+args.dataset+"_data.pkl")
    device = torch.device('cuda')
    data = data.to(device)
  
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
    y_all = np.logical_or(data.y, data.y_te)
    mask_all = np.logical_or(data.mask, data.mask_te)
     
    if cancerType=='pan-cancer':
        data.x = data.x[:,:48]
    else:
        cancerType_dict = {
                          'kirc':[0,16,32],
                          'brca':[1,17,33],
                          'prad':[3,19,35],
                          'stad':[4,20,36],
                          'hnsc':[5,21,37],
                          'luad':[6,22,38],
                          'thca':[7,23,39],
                          'blca':[8,24,40],
                          'esca':[9,25,41],
                          'lihc':[10,26,42],
                          'ucec':[11,27,43],
                          'coad':[12,28,44],
                          'lusc':[13,29,45],
                          'cesc':[14,30,46],
                          'kirp':[15,31,47]
                                  }
        data.x = data.x[:, cancerType_dict[cancerType]]
    print(data.x)

    #node2VEC feature
    dataz = torch.load(dataPath+"Str_feature.pkl")
    dataz=dataz.to(device)
    data.x = torch.cat((data.x,dataz),1)#64D feature
    ppiAdj = torch.load(dataPath+'ppi.pkl')
    ppiAdj_self = torch.load(dataPath+'ppi_selfloop.pkl')
    pathAdj = torch.load(dataPath+'pathway_SimMatrix.pkl')
    goAdj = torch.load(dataPath+'GO_SimMatrix.pkl')
    pos = ppiAdj_self.to_dense()

    if args.dataset =='CPDB':
        with open(dataPath+"k_sets.pkl", 'rb') as handle:
            k_sets = pickle.load(handle)
    else:
        k_sets = torch.load(dataPath+"k_sets.pkl")
    
    AUC = np.zeros(shape=(cross_val, 5))
    AUPR = np.zeros(shape=(cross_val, 5))
    train_time = t()
    #pan-cancer
    print("---------Pan-cancer Train begin--------")
    for i in range(cross_val):
        for cv_run in range(5):
            print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
            start = t()
            y_tr, y_te, tr_mask, te_mask = k_sets[i][cv_run]
            gcn = GCN(64,300,100).to(device)
            model = MNGCL( gnn=gcn,
                           pos=pos,
                           tau=tau,
                           gnn_outsize=100,
                           projection_hidden_size=300,
                           projection_size=100
                       ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            for train_epoch in range(1,EPOCH):
                loss = train(tr_mask)
                AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask,te_mask)      
                print(f'(T) | Epoch={train_epoch:03d}, loss={loss:.4f},AUC={AUC[i][cv_run]:.4f}, AUPR={AUPR[i][cv_run]:.4f}')
            np.savetxt("./AUC.txt", AUC, delimiter="\t")
            np.savetxt("./AUPR.txt", AUPR, delimiter="\t")
            now = t()
            print("this cv_run spend {:.1f}  s".format(now - start))
        print(AUC[i].mean())

    #specific cancer
    '''print("---------"+cancerType+" cancer Train begin--------")
    path = dataPath+"Specific cancer/"
    for i in range(cross_val):
        label, label_pos, label_neg = load_label_single(path)
        random.shuffle(label_pos)
        random.shuffle(label_neg)
        l = len(label)
        l1 = int(len(label_pos)/5)
        l2 = int(len(label_neg)/5)
        Y = label
        for cv_run in range(5):
            print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
            start = t()
            tr_mask, te_mask = sample_division_single(label_pos, label_neg, l, l1, l2, cv_run)
            gcn = GCN(19,150,50).to(device)
            model = MNGCL(gnn=gcn,
                           pos=pos,
                           tau=tau,
                           gnn_outsize=50,
                           projection_hidden_size=150,
                           projection_size=50
                       ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            for train_epoch in range(1,EPOCH):
                loss = train(tr_mask)
                AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask,te_mask)      
                print(f'(T) | Epoch={train_epoch:03d}, loss={loss:.4f},AUC={AUC[i][cv_run]:.4f}, AUPR={AUPR[i][cv_run]:.4f}')
            
            np.savetxt("./AUC.txt", AUC, delimiter="\t")
            np.savetxt("./AUPR.txt", AUPR, delimiter="\t")
            now = t()
           
            print("this cv_run spend {:.2f}  s".format(now - start))
        print(AUC[i].mean())
        print(AUPR[i].mean())'''



