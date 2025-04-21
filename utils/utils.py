import torch
import random
import math
import psutil
import numpy as np
from PIL import Image
from torch.optim.optimizer import Optimizer

class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, device='cpu'):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().to(device) for p in params]
        self.device = device
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) 
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                fp32_p.addcdiv_(exp_avg, denom, value=-step_size)
                p.data = fp32_p.half()

        return loss
    
class LogSaver():
    def __init__(self, save_path="log.txt"):
        self.save_path = save_path
    def save_log(self, data, file=None, print_data=True):
        """Function print and store the logs."""
        if(print_data):
            print(str(data))
        if file is None:
            file = self.save_path
        with open(str(file), 'a') as f:
            f.write(str(data))
            f.write("\n")

def set_seed(random_state):
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(random_state) # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).
    torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm, 
                                                # possibly at the cost of reduced performance 
                                                # (the algorithm itself may be nondeterministic).
    torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm, 
                                                # but may slow down performance.
                                                # It will not guarantee that your training process is deterministic 
                                                # if you are using other libraries that may use nondeterministic algorithms 

    g = torch.Generator()
    g.manual_seed(random_state)


def join_img_maks(image, mask, alpha=0.2):
    slice_mask = np.stack((mask,)*3, axis=-1)
    slice_mask = np.where(slice_mask == [1,1,1], [1,0,0], [0,0,0])
    slice_mask = np.array(slice_mask, dtype=float)
    slice = np.stack((image,)*3, axis=-1)
    return (slice+slice_mask*alpha * 255).astype(np.uint8)


def put_color_mask(mask, palette, alpha_value=128):
    """Function to put pallete of colors in numpy array masks. Inputs: numpy
    image mask, pallete of colors, and tranparency level (alpha_value). It
    returns a tuple with PIL Images with and without transparency."""
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')
    new_mask_no_alpha = new_mask.copy()
    new_mask.putalpha(alpha_value)
    return new_mask_no_alpha, new_mask


def memory_usage(detailed_output=False):
    """Auxiliary function to return a string containing information about the memory usage."""
    total = psutil.virtual_memory().total / (1024.0 ** 3)
    percent = psutil.virtual_memory().percent
    str_memory = f'Total memory [GB]: {total}, percentage used: {percent}%'
    if detailed_output:
        available = psutil.virtual_memory().available / (1024.0 ** 3)
        used = psutil.virtual_memory().used / (1024.0 ** 3)
        free = psutil.virtual_memory().free / (1024.0 ** 3)
        active = psutil.virtual_memory().active / (1024.0 ** 3)
        inactive = psutil.virtual_memory().inactive / (1024.0 ** 3)
        str_memory = str_memory + f'. Other info [GB]: Available: {available}, Used: {used}, Free: {free}, Active: {active}, Inactive: {inactive}'
    return str_memory