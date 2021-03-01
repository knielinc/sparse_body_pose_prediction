import torch
from torch import nn
from Helpers.perceptual_loss.st_gcn import *

class GCNLoss(nn.Module):
    def __init__(self,opt):
        super(GCNLoss, self).__init__()
        dict_path=opt.pretrain_GCN
        graph_args={"layout": 'openpose',"strategy": 'spatial'}
        self.gcn = Model(2,16,graph_args,edge_importance_weighting=True).cuda()
        self.gcn.load_state_dict(torch.load(dict_path))
        self.gcn.eval()
        self.criterion = nn.L1Loss()
        self.weights = [20.0 ,5.0 ,1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  #10 output

    def forward(self, x, y):
        x_gcn, y_gcn = self.gcn.extract_feature(x), self.gcn.extract_feature(y)
        loss = 0
        for i in range(len(x_gcn)):
            loss_state = self.weights[i] * self.criterion(x_gcn[i], y_gcn[i].detach())
            #print("VGG_loss "+ str(i),loss_state.item())
            loss += loss_state
        return loss

