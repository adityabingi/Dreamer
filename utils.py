import os
import pickle
import torch
import numpy as np 
import moviepy.editor as mpy

import matplotlib.pyplot as plt 
from typing import Iterable
from torch.nn import Module
from tensorboardX import SummaryWriter



def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
          output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):

        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

class Logger:

    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, step):
        for key, value in scalar_dict.items():
            print('{} : {}'.format(key, value))
            self.log_scalar(value, key, step)
        self.dump_scalars_to_pickle(scalar_dict, step)

    def log_videos(self, videos, step, max_videos_to_save=1, fps=20, video_title='video'):

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, videos.shape[0]])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

            clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
            new_video_title = video_title+'{}_{}'.format(step, i) + '.gif'
            filename = os.path.join(self._log_dir, new_video_title)
            video.write_gif(filename, fps =fps)


    def dump_scalars_to_pickle(self, metrics, step, log_title=None):
        log_path = os.path.join(self._log_dir, "scalar_data.pkl" if log_title is None else log_title)
        with open(log_path, 'ab') as f:
            pickle.dump({'step': step, **dict(metrics)}, f)

    def flush(self):
        self._summ_writer.flush()

def compute_return(rewards, values, discounts, td_lam, last_value):

    next_values = torch.cat([values[1:], last_value.unsqueeze(0)],0)  
    targets = rewards + discounts * next_values * (1-td_lam)
    rets =[]
    last_rew = last_value

    for t in range(rewards.shape[0]-1, -1, -1):
        last_rew = targets[t] + discounts[t] * td_lam *(last_rew)
        rets.append(last_rew)

    returns = torch.flip(torch.stack(rets), [0])
    return returns
