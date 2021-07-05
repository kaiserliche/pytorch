import torch
import torch.optim as optim
import numpy as np
from torch.optim.optimizer import required
from collections import defaultdict
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import math
from torch import Tensor
from typing import List, Optional
from torch.autograd.functional import jacobian

# def grad_calc(x,v,delta):
#     return (1-delta)*x+(delta*v)

class AFW(optim.Optimizer):
    """
    Implements Accelerated Frank Wolfe: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9457128

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        theta (float) : "see in the paper"
        iter (int): number of iterations to perform
        epsilon (float): minimum error

    Example:
        >>> optimizer = AFW(model.parameters(),theta = 0, iter=1000, eps = 1e-5)
    """

    def __init__(self, params, theta = 0, iter = 1000, eps=1e-5):

        defaults = dict(theta = theta ,iter = iter, eps = eps)
        self.eps = eps
        super(AFW, self).__init__(params, defaults)
        

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ##code
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state=self.state[p]
                # print(self.state)
                if len(state) == 0:
                  state['step'] = 0
                  state['theta_k'] = torch.zeros_like(p.data)
                  state['v_k']=torch.clone(p.data)
                #   state['y_k'] = Variable(p.data , requires_grad = True)
                step=state['step']
                theta_k=state['theta_k']
                v_k=state['v_k']
                y_k = []
                #update expression

                delta_k=torch.tensor(2/(step+3))
                # y_k=jacobian(grad_calc,(p.data,v_k,delta_k))
                delta_k=delta_k.item()
                # print(torch.mul(p.grad.data,y_k[0]).size())
                y_k = torch.add((1-delta_k)*p.data, delta_k*v_k)
                state['theta_k']= torch.add((1-delta_k)*theta_k,delta_k*(y_k.grad))
                state['v_k']= -(20/(2 * torch.linalg.norm(theta_k)) * theta_k
                p.data.add_((1-delta_k)*p.data,delta_k*state['v_k'])
                state['step']+=1          
        
    return loss