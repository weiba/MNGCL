import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, out):
            super(GCN, self).__init__()
            self.gc1 = ChebConv(nfeat, nhid, K=2, normalization="sym")
            self.gc2 = ChebConv(nhid, out, K=2, normalization="sym")
  

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
            x1= self.gc1(x, adj)
            x1 =  F.relu(x1)
            x2 = self.gc2(x1, adj)
            x2 =  F.relu(x2)
            return  x2
            
            
