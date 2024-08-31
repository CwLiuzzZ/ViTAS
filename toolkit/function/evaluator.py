import torch


def calcu_EPE(pre,gt):
    ans = torch.mean(torch.abs(gt - pre))
    return ans

def calcu_PEP(pre,gt,thr=1):
    abs_diff = torch.abs(gt - pre)
    num_base = torch.sum(abs_diff>=0)
    num_error = torch.sum(abs_diff>=thr)
    ans = num_error/num_base
    return ans

def calcu_D1all(pre,gt):
    abs_diff = torch.abs(gt - pre)
    num_base = torch.sum(abs_diff>=0)
    error = torch.ones(abs_diff.shape)
    error[abs_diff < gt*0.05] = 0
    error[abs_diff < 3] = 0
    num_error = torch.sum(error)
    ans = num_error/num_base
    return ans