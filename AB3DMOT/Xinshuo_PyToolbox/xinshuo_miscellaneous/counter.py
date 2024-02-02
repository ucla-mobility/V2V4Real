# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

import time, sys, numpy as np

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')

def get_iso_timestring(mode='normal'):
    if mode == 'file':
        ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
    elif mode == 'normal':
        ISOTIMEFORMAT='%Y-%m-%d %X'
    elif mode == 'short':
        ISOTIMEFORMAT='%Y%m%d'            
    else:
        assert False, 'mode error for iso time string'

    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))

class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0    
    self.list = list()
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count  
    self.list.append(val)

class LossRecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch   = total_epoch
        self.current_epoch = 0
        self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
        self.epoch_losses  = self.epoch_losses + sys.float_info.max

    def update(self, train_loss, idx, val_loss=None):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        if val_loss is not None:
            self.epoch_losses[idx, 1] = val_loss
        self.current_epoch = idx + 1
  
    def min_loss(self, Train=True):
        if Train:
            idx = np.argmin(self.epoch_losses[:self.current_epoch, 0])
            return idx, self.epoch_losses[idx, 0]
        else:
            idx = np.argmin(self.epoch_losses[:self.current_epoch, 1])
            if self.epoch_losses[idx, 1] >= sys.float_info.max / 10:
                return idx, -1.
            else:
                return idx, self.epoch_losses[idx, 1]
