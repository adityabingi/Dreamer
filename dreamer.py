import os
import random
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.distributions as distributions

from collections import OrderedDict

import env_wrapper
from replay_buffer import ReplayBuffer
from models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from utils import *

os.environ['MUJOCO_GL'] = 'egl'

def make_env(args):

    env = env_wrapper.DeepMindControl(args.env, args.seed)
    env = env_wrapper.ActionRepeat(env, args.action_repeat)
    env = env_wrapper.NormalizeActions(env)
    env = env_wrapper.TimeLimit(env, args.time_limit / args.action_repeat)
    #env = env_wrapper.RewardObs(env)
    return env

def preprocess_obs(obs):
    obs = obs.to(torch.float32)/255.0 - 0.5
    return obs

class Dreamer:

    def __init__(self, args, obs_shape, action_size, device, restore=False):

        self.args = args
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device
        self.restore = args.restore
        self.restore_path = args.checkpoint_path
        self.data_buffer = ReplayBuffer(self.args.buffer_size, self.obs_shape, self.action_size,
                                                    self.args.train_seq_len, self.args.batch_size)

        self._build_model(restore=self.restore)

    def _build_model(self, restore):

        self.rssm = RSSM(
                    action_size =self.action_size,
                    stoch_size = self.args.stoch_size,
                    deter_size = self.args.deter_size,
                    hidden_size = self.args.deter_size,
                    obs_embed_size = self.args.obs_embed_size,
                    activation =self.args.dense_activation_function).to(self.device)

        self.actor = ActionDecoder(
                     action_size = self.action_size,
                     stoch_size = self.args.stoch_size,
                     deter_size = self.args.deter_size,
                     units = self.args.num_units,
                     n_layers=4,
                     activation=self.args.dense_activation_function).to(self.device)
        self.obs_encoder  = ConvEncoder(
                            input_shape= self.obs_shape,
                            embed_size = self.args.obs_embed_size,
                            activation =self.args.cnn_activation_function).to(self.device)
        self.obs_decoder  = ConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape=self.obs_shape,
                            activation = self.args.cnn_activation_function).to(self.device)
        self.reward_model = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units=self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal').to(self.device)
        self.value_model  = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal').to(self.device) 
        if self.args.use_disc_model:  
          self.discount_model = DenseDecoder(
                                stoch_size = self.args.stoch_size,
                                deter_size = self.args.deter_size,
                                output_shape = (1,),
                                n_layers = 2,
                                units=self.args.num_units,
                                activation= self.args.dense_activation_function,
                                dist = 'binary').to(self.device)
        
        if self.args.use_disc_model:
          self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters()) + list(self.discount_model.parameters())
        else:
          self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters())
    
        self.world_model_opt = optim.Adam(self.world_model_params, self.args.model_learning_rate)
        self.value_opt = optim.Adam(self.value_model.parameters(), self.args.value_learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), self.args.actor_learning_rate)

        if self.args.use_disc_model:
          self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model, self.discount_model]
        else:
          self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model]
        self.value_modules = [self.value_model]
        self.actor_modules = [self.actor]

        if restore:
            self.restore_checkpoint(self.restore_path)

    def world_model_loss(self, obs, acs, rews, nonterms):

        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:])
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len-1)
        features = torch.cat([self.posterior['stoch'], self.posterior['deter']], dim=-1)
        rew_dist = self.reward_model(features)
        obs_dist = self.obs_decoder(features)
        if self.args.use_disc_model:
          disc_dist = self.discount_model(features)

        prior_dist = self.rssm.get_dist(prior['mean'], prior['std'])
        post_dist = self.rssm.get_dist(self.posterior['mean'], self.posterior['std'])

        if self.args.algo == 'Dreamerv2':
            post_no_grad = self.rssm.detach_state(self.posterior)
            prior_no_grad = self.rssm.detach_state(prior)
            post_mean_no_grad, post_std_no_grad = post_no_grad['mean'], post_no_grad['std']
            prior_mean_no_grad, prior_std_no_grad = prior_no_grad['mean'], prior_no_grad['std']
            
            kl_loss = self.args.kl_alpha *(torch.mean(distributions.kl.kl_divergence(
                               self.rssm.get_dist(post_mean_no_grad, post_std_no_grad), prior_dist)))
            kl_loss += (1-self.args.kl_alpha) * (torch.mean(distributions.kl.kl_divergence(
                               post_dist, self.rssm.get_dist(prior_mean_no_grad, prior_std_no_grad))))
        else:
            kl_loss = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))
            kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), self.args.free_nats))

        obs_loss = -torch.mean(obs_dist.log_prob(obs[1:])) 
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        if self.args.use_disc_model:
          disc_loss = -torch.mean(disc_dist.log_prob(nonterms[:-1]))

        if self.args.use_disc_model:
          model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss + self.args.disc_loss_coeff * disc_loss
        else:
          model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss 
        
        return model_loss

    def actor_loss(self):

        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states = self.rssm.imagine_rollout(self.actor, posterior, self.args.imagine_horizon)

        self.imag_feat = torch.cat([imag_states['stoch'], imag_states['deter']], dim=-1)

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.value_model(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts =  self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1],discounts[:-1] \
                                         ,self.args.td_lambda, imag_vals[-1])

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        return actor_loss

    def value_loss(self):

        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            discount   = self.discounts.detach()
            value_targ = self.returns.detach()

        value_dist = self.value_model(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))
        
        return value_loss

    def train_one_batch(self):

        obs, acs, rews, terms = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)

        model_loss = self.world_model_loss(obs, acs, rews, nonterms)
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        self.world_model_opt.step()

        actor_loss = self.actor_loss()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
        self.actor_opt.step()

        value_loss = self.value_loss()
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm)
        self.value_opt.step()

        return model_loss.item(), actor_loss.item(), value_loss.item()

    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):

        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = torch.cat([posterior['stoch'], posterior['deter']], dim=-1)
        action = self.actor(features, deter=not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        return  posterior, action

    def act_and_collect_data(self, env, collect_steps):

        obs = env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action = self.act_with_world_model(obs, prev_state, prev_action, explore=True)
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, done)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

        return np.array(episode_rewards)

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = torch.zeros(1, self.action_size).to(self.device)

            while not done:
                with torch.no_grad():
                    posterior, action = self.act_with_world_model(obs, prev_state, prev_action)
                action = action[0].cpu().numpy()
                next_obs, rew, done, _ = env.step(action)
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1,2,0).copy())
                obs = next_obs
        return episode_rew, np.array(video_images[:self.args.max_videos_to_save])

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)
            
            self.data_buffer.add(obs, action, rew, done)
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i!= seed_steps-1:
                    seed_episode_rews.append(0.0)
                done=False  
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_path):

        torch.save(
            {'rssm' : self.rssm.state_dict(),
            'actor': self.actor.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.args.use_disc_model else None,
            'actor_optimizer': self.actor_opt.state_dict(),
            'value_optimizer': self.value_opt.state_dict(),
            'world_model_optimizer': self.world_model_opt.state_dict(),}, save_path)

    def restore_checkpoint(self, ckpt_path):

        checkpoint = torch.load(ckpt_path)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        if self.args.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])

        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.value_opt.load_state_dict(checkpoint['value_optimizer'])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='walker_walk', help='Control Suite environment')
    parser.add_argument('--algo', type=str, default='Dreamerv1', choices=['Dreamerv1', 'Dreamerv2'], help='choosing algorithm')
    parser.add_argument('--exp-name', type=str, default='lr1e-3', help='name of experiment for logging')
    parser.add_argument('--train', action='store_true', help='trains the model')
    parser.add_argument('--evaluate', action='store_true', help='tests the model')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--no-gpu', action='store_true', help="GPUs aren't used if passed true")
    # Data parameters
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max episode length')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--time-limit', type=int, default=1000, help='time limit') # Environment TimeLimit
    # Models parameters
    parser.add_argument('--cnn-activation-function', type=str, default='relu', help='Model activation function for a convolution layer')
    parser.add_argument('--dense-activation-function', type=str, default='elu', help='Model activation function a dense layer')
    parser.add_argument('--obs-embed-size', type=int, default=1024, help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--num-units', type=int, default=400, help='num hidden units for reward/value/discount models')
    parser.add_argument('--deter-size', type=int, default=200, help='GRU hidden size and deterministic belief size')
    parser.add_argument('--stoch-size', type=int, default=30, help='Stochastic State/latent size')
    # Actor Exploration Parameters
    parser.add_argument('--action-repeat', type=int, default=2, help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, help='Action noise')
    # Training parameters
    parser.add_argument('--total_steps', type=int, default=5e6, help='total number of training steps')
    parser.add_argument('--seed-steps', type=int, default=5000, help='seed episodes')
    parser.add_argument('--update-steps', type=int, default=100, help='num of train update steps per iter')
    parser.add_argument('--collect-steps', type=int, default=1000, help='actor collect steps per 1 train iter')
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--train-seq-len', type=int, default=50, help='sequence length for training world model')
    parser.add_argument('--imagine-horizon', type=int, default=15, help='Latent imagination horizon')
    parser.add_argument('--use-disc-model', action='store_true', help='whether to use discount model' )
    # Coeffecients and constants
    parser.add_argument('--free-nats', type=float, default=3, help='free nats')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor for actor critic')
    parser.add_argument('--td-lambda', type=float, default=0.95, help='discount rate to compute return')
    parser.add_argument('--kl-loss-coeff', type=float, default=1.0, help='weightage for kl_loss of model')
    parser.add_argument('--kl-alpha', type=float, default=0.8, help='kl balancing weight; used for Dreamerv2')
    parser.add_argument('--disc-loss-coeff', type=float, default=10.0, help='weightage of discount model')
    
    # Optimizer Parameters
    parser.add_argument('--model_learning-rate', type=float, default=6e-4, help='World Model Learning rate') 
    parser.add_argument('--actor_learning-rate', type=float, default=8e-5, help='Actor Learning rate') 
    parser.add_argument('--value_learning-rate', type=float, default=8e-5, help='Value Model Learning rate')
    parser.add_argument('--adam-epsilon', type=float, default=1e-7, help='Adam optimizer epsilon value') 
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, help='Gradient clipping norm')
    # Eval parameters
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--test-interval', type=int, default=10000, help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    # saving and checkpoint parameters
    parser.add_argument('--scalar-freq', type=int, default=1e3, help='scalar logging freq')
    parser.add_argument('--log-video-freq', type=int, default=-1, help='video logging frequency')
    parser.add_argument('--max-videos-to-save', type=int, default=2, help='max_videos for saving')
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Load model checkpoint')
    parser.add_argument('--restore', action='store_true', help='restores model from checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')


    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.env + '_' + args.algo + '_' + args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    train_env = make_env(args)
    test_env  = make_env(args)
    obs_shape = train_env.observation_space['image'].shape
    action_size = train_env.action_space.shape[0]
    dreamer = Dreamer(args, obs_shape, action_size, device, args.restore)

    logger = Logger(logdir)

    if args.train:
        initial_logs = OrderedDict()
        seed_episode_rews = dreamer.collect_random_episodes(train_env, args.seed_steps//args.action_repeat)
        global_step = dreamer.data_buffer.steps * args.action_repeat

        # without loss of generality intial rews for both train and eval are assumed same
        initial_logs.update({
            'train_avg_reward':np.mean(seed_episode_rews),
            'train_max_reward': np.max(seed_episode_rews),
            'train_min_reward': np.min(seed_episode_rews),
            'train_std_reward':np.std(seed_episode_rews),
            'eval_avg_reward': np.mean(seed_episode_rews),
            'eval_max_reward': np.max(seed_episode_rews),
            'eval_min_reward': np.min(seed_episode_rews),
            'eval_std_reward':np.std(seed_episode_rews),
            })

        logger.log_scalars(initial_logs, step=0)
        logger.flush()

        while global_step <= args.total_steps:

            print("##################################")
            print(f"At global step {global_step}")

            logs = OrderedDict()

            for _ in range(args.update_steps):
                model_loss, actor_loss, value_loss = dreamer.train_one_batch()
    
            train_rews = dreamer.act_and_collect_data(train_env, args.collect_steps//args.action_repeat)

            logs.update({
                'model_loss' : model_loss,
                'actor_loss': actor_loss,
                'value_loss': value_loss,
                'train_avg_reward':np.mean(train_rews),
                'train_max_reward': np.max(train_rews),
                'train_min_reward': np.min(train_rews),
                'train_std_reward':np.std(train_rews),
            })

            if global_step % args.test_interval == 0:
                episode_rews, video_images = dreamer.evaluate(test_env, args.test_episodes)

                logs.update({
                    'eval_avg_reward':np.mean(episode_rews),
                    'eval_max_reward': np.max(episode_rews),
                    'eval_min_reward': np.min(episode_rews),
                    'eval_std_reward':np.std(episode_rews),
                })
            
            logger.log_scalars(logs, global_step)

            if global_step % args.log_video_freq ==0 and args.log_video_freq != -1 and len(video_images[0])!=0:
                logger.log_video(video_images, global_step, args.max_videos_to_save)

            if global_step % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(logdir, 'ckpts/')
                if not (os.path.exists(ckpt_dir)):
                    os.makedirs(ckpt_dir)
                dreamer.save(os.path.join(ckpt_dir,  f'{global_step}_ckpt.pt'))

            global_step = dreamer.data_buffer.steps * args.action_repeat
            logger.flush()

    elif args.evaluate:
        logs = OrderedDict()
        episode_rews, video_images = dreamer.evaluate(test_env, args.test_episodes, render=True)

        logs.update({
            'test_avg_reward':np.mean(episode_rews),
            'test_max_reward': np.max(episode_rews),
            'test_min_reward': np.min(episode_rews),
            'test_std_reward':np.std(episode_rews),
        })

        logger.dump_scalars_to_pickle(logs, 0, log_title='test_scalars.pkl')
        logger.log_videos(video_images, 0, max_videos_to_save=args.max_videos_to_save)

if __name__ == '__main__':
    main()
