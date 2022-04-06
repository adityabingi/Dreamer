import torch
import numpy as np


class ReplayBuffer:

    def __init__(self, size, obs_shape, action_size, seq_len, batch_size):

        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8) 
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32) 
        self.terminals = np.empty((size,), dtype=np.float32)
        self.steps, self.episodes = 0, 0
    
    def add(self, obs, ac, rew, done):

        self.observations[self.idx] = obs['image']
        self.actions[self.idx] = ac
        self.rewards[self.idx] = rew
        self.terminals[self.idx] = done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1 
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, L):

        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = self.observations[vec_idxs]
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.terminals[vec_idxs].reshape(L, n)

    def sample(self):
        n = self.batch_size
        l = self.seq_len
        obs,acs,rews,terms= self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        return obs,acs,rews,terms
