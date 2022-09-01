

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(56,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )
        
        self.Decoder = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,14),
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x

class ShortSkipConnection(nn.Module):
    def __init__(self):
        super(ShortSkipConnection, self).__init__()
        
        self.ln = nn.LayerNorm(7000)
        self.ln1 = nn.LayerNorm(5000)
        self.ln2 = nn.LayerNorm(3000)
        self.upblock1 = nn.Sequential(nn.Linear(56, 1000),nn.BatchNorm1d(1000), nn.GELU())
        self.upblock2 = nn.Sequential(nn.Linear(1000,3000),nn.BatchNorm1d(3000),nn.GELU())
        self.upblock3 = nn.Sequential(nn.Linear(3000,5000),nn.BatchNorm1d(5000), nn.GELU())
        self.upblock4 = nn.Sequential(nn.Linear(5000,7000),nn.BatchNorm1d(7000),nn.GELU())

        self.downblock1 = nn.Sequential(nn.Linear(7000, 5000),nn.BatchNorm1d(5000),nn.GELU())
        self.downblock2 = nn.Sequential(nn.Linear(5000, 3000),nn.BatchNorm1d(3000),nn.GELU())
        self.downblock3 = nn.Sequential(nn.Linear(3000, 1000),nn.BatchNorm1d(1000),nn.GELU())
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),nn.BatchNorm1d(300),nn.GELU())
        
        self.fclayer = nn.Sequential(nn.Linear(300,14))
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(skipblock3)
        
        output = self.fclayer(downblock4_out)
        
        return output



class NRMSELoss(torch.nn.Module):
    def __init__(self):
        super(NRMSELoss,self).__init__()

    def forward(self, gt, preds):
        criterion = nn.MSELoss()
        all_nrmse = torch.zeros(14)
        for idx in range(14):
            rmse = torch.sqrt(criterion(preds[:,idx], gt[:,idx]))
            nrmse = rmse / torch.mean(torch.abs(gt[:,idx]))
            all_nrmse[idx] = nrmse
        score = 1.2 * torch.sum(all_nrmse[:8]) + 1.0 * torch.sum(all_nrmse[8:14])
        return score
