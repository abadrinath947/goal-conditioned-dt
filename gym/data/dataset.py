"""Builds the data modules used to train the policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple, Union

from d4rl import offline_env
import gym
import numpy as np
import torch
from torch.utils import data

max_num_workers = 16
device = 'cuda'

class D4RLIterableDataset(data.IterableDataset):
    """Used for goal-conditioned learning in D4RL."""

    def __init__(
        self,
        trajectories: List[Dict],
        p_sample: Optional[np.ndarray] = None,
        epoch_size: int = 2450000,
        index_batch_size: int = 64,
        goal_columns: Optional[Union[Tuple[int], List[int], np.ndarray]] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """Initializes the dataset.

        Args:
            observations: The observations for the dataset.
            actions: The actions for the dataset.
            dones: The dones for the dataset.
            epoch_size: For PyTorch Lightning to count epochs.
            index_batch_size: This has no effect on the functionality of the dataset,
                but it is used internally as the batch size to fetch random indices.
            goal_columns: If not None, then only use these columns of the
                observation_space for the goal conditioning.
        """
        super().__init__()

        self.trajectories = trajectories
        self.lengths = np.array([len(self.trajectories[i]['observations']) for i in range(len(trajectories))])
        self.epoch_size = epoch_size
        self.index_batch_size = index_batch_size
        self.goal_columns = goal_columns
        self.config = config if config is not None else defaultdict(bool)
        self.config.update(kwargs)

    def _sample_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Credit to Dibya Ghosh's GCSL codebase for the logic in the following block:
        # https://github.com/dibyaghosh/gcsl/blob/
        # cfae5609cee79e5a2228fb7653451023c41a64cb/gcsl/algo/buffer.py#L78
        trajectory_indices = np.random.choice(len(self.trajectories), self.index_batch_size, 
                                              p = self.lengths / sum(self.lengths))
        lengths = np.array([len(self.trajectories[i]['observations']) for i in trajectory_indices])
        proportional_indices_1 = np.random.rand(self.index_batch_size)
        proportional_indices_2 = np.random.rand(self.index_batch_size)
        time_indices_1 = np.floor(
            proportional_indices_1 * lengths,
        ).astype(int)
        time_indices_2 = np.floor(
            proportional_indices_2 * lengths,
        ).astype(int)

        start_indices = np.minimum(
            time_indices_1,
            time_indices_2,
        )
        goal_indices = np.maximum(
            time_indices_1,
            time_indices_2,
        )

        return trajectory_indices, start_indices, goal_indices

    def _fetch_trajectories(self, trajectory_indices, start_indices, goal_indices):
        K, max_ep_len = self.config['K'], self.config['max_ep_len']
        goal_dim, state_dim, act_dim = self.config['goal_dim'], self.config['state_dim'], self.config['act_dim']
        state_mean, state_std = self.config['state_mean'], self.config['state_std']
        trajs = [self.trajectories[i] for i in trajectory_indices] 
        s, a, r, d, rtg, timesteps, mask, g = [], [], [], [], [], [], [], []
        for (si, gi), traj in zip(zip(start_indices, goal_indices), trajs):
            # get sequences from dataset
            s.append(traj['observations'][si:si + K].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + K].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + K].reshape(1, -1, 1))
            d.append(traj['terminals'][si:si + K].reshape(1, -1))
            if self.config['goal_conditioned']:
                g.append(np.hstack([traj['observations'][gi, self.goal_columns].reshape(1, -1, goal_dim)] * s[-1].shape[1]))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, K - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, K - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), r[-1]], axis=1)
            if self.config['goal_conditioned']:
                g[-1] = (g[-1] - state_mean[:goal_dim]) / state_std[:goal_dim]
                g[-1] = np.concatenate([np.zeros((1, K - tlen, goal_dim)), g[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, K - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, K - tlen, 1)), rtg[-1]], axis=1) #/ scale
            timesteps[-1] = np.concatenate([np.zeros((1, K - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, K - tlen)), np.ones((1, tlen))], axis=1))
            
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        if self.config['goal_conditioned']:
            g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        if self.config['goal_conditioned']:
            return s, a, r, d, g, timesteps, mask
        return s, a, r, d, rtg, timesteps, mask

    def _sample_batch(self, batch_size = None) -> Tuple:
        trajectory_indices, start_indices, goal_indices = self._sample_indices()
        s, a, r, d, cond, timesteps, mask = self._fetch_trajectories(trajectory_indices, 
                                                                     start_indices, 
                                                                     goal_indices)
        return s, a, r, d, cond, timesteps, mask

    def __iter__(self) -> Iterator[Tuple[torch.tensor, torch.tensor]]:
        """Yield each training example."""
        examples_yielded = 0
        while examples_yielded < self.epoch_size:
            yield self._sample_batch()
            examples_yielded += self.index_batch_size

    def __len__(self) -> int:
        """The number of examples in an epoch. Used by the trainer to count epochs."""
        return self.epoch_size


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

