import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失函数。

    该损失函数假设目标值(target)服从一个高斯分布（正态分布），
    而模型的任务是预测这个分布的均值(mean)和方差(variance)。
    它适用于回归任务，特别是当需要模型量化其预测的不确定性时。

    Args:
        **kwargs: 传入 torch.nn.GaussianNLLLoss 的额外参数, 
                  例如 full=True, eps=1e-6, reduction='mean'。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.nll_loss = nn.GaussianNLLLoss(**kwargs)

    def forward(self, pred_mean, target, pred_var):
        """
        前向传播计算损失。

        :param pred_mean: 模型预测的均值张量，形状应与 target 一致。
        :param target: 真实的目标值张量。
        :param pred_var: 模型预测的方差张量，形状应与 target 一致。方差必须为正。
        :return: 计算出的标量损失值。
        """
        # PyTorch 的 GaussianNLLLoss 要求输入(input)是预测的均值
        return self.nll_loss(pred_mean, target, pred_var)