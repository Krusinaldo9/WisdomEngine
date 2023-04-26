import torch
import torch.nn as nn
from scipy.stats import beta

def assign_loss_function(loss_name):

    if loss_name == 'ExpMSE':
        return ExpMSE()
    elif loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'Huber':
        return nn.HuberLoss()
    elif loss_name == 'Inv_MSE':
        return Inv_MSE()
    else:
        print('Loss Function', loss_name, 'is not implemented. ExpMSE is used instead')
        return ExpMSE()


class ExpMSE(nn.Module):

    def __init__(self):
        super(ExpMSE, self).__init__()

    def forward(self, inputs, targets):

        mins = torch.minimum(inputs, targets)
        maxs = torch.maximum(inputs, targets)

        mins_exp = torch.exp(mins)
        maxs_exp = torch.exp(maxs)

        tmp_1 = torch.abs(mins_exp / maxs_exp)

        tmp = torch.abs(tmp_1 - 1)

        mse = (inputs - targets) ** 2

        loss = torch.mean(tmp + mse)
        return loss

class Inv_MSE(nn.Module):

    def __init__(self):
        super(Inv_MSE, self).__init__()
        self.inverse = False

    def forward(self, inputs, targets):

        mse = (inputs - targets) ** 2

        if self.training:
            self.inverse = not self.inverse
            b = beta.rvs(1, 2, size=1)[0]
            q = torch.quantile(mse, b)

            if self.inverse:
                mse[mse < q] = 0
            else:
                mse[mse > q] = 0

        if self.inverse:
            return torch.mean(mse)
        else:
            return torch.mean(mse)

