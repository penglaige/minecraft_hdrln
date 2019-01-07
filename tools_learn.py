"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
from utils.minecraft_wrappers import ENV
from logger import Logger
import time

import MalmoPython

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

#CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# dqn_learner
class DQNAgent(object):
    """A dqn learner agent."""

    def __init__(self,env,q_func,optimizer_spec,num_actions,
        exploration=LinearSchedule(200000, 0.1),
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        img_h=84,
        img_w=84,
        img_c=1,
        target_update_freq=10000,
        double_dqn=False,
        dueling_dqn=False):
        """Run Deep Q-learning algorithm.
        You can specify your own convnet using q_func.
        All schedules are w.r.t. total number of steps taken in the environment.
        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        env_id: string
            gym environment id for model saving.
        q_func: function
            Model to use for computing the q function.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        stopping_criterion: (env, t) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        """
        self.logger = Logger('./logs')
        self.env = env
        self.q_func = q_func
        self.optimizer_spec = optimizer_spec
        self.exploration = exploration
        self.stopping_criterion = stopping_criterion
        self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.input_shape = (self.img_h, self.img_w, self.frame_history_len * self.img_c)
        self.in_channels = self.input_shape[2]

        # define Q target and Q
        self.Q = self.q_func(self.in_channels, self.num_actions).type(dtype)
        self.Q_target = self.q_func(self.in_channels, self.num_actions).type(dtype)

        #initialize optimizer
        self.optimizer = self.optimizer_spec.constructor(self.Q.parameters(), **self.optimizer_spec.kwargs)

        #create replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len,self.img_w,self.img_h,"scale")

        ###### RUN SETTING ####
        self.t = 0
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = None
        self.Log_Every_N_Steps = 10
        self.Save_Model_Every_N_Steps = 100
        #########################

    #Set the logger
    def to_np(self,x):
        return x.data.cpu().numpy() #???

    def run(self, agent_host):
        #run many times
        if self.last_obs is None:
            self.last_obs = self.env.reset()

        for t in itertools.count():
            ### 1. Check stopping criterion
            if self.stopping_criterion is not None and self.stopping_criterion(agent_host, t):
                if self.env.canset():
                    self.last_obs = self.env.reset()
                else:
                    break

            ### 2. Step the agent and store the transition
            # store last frame, returned idx used later
            #if self.last_obs is None:
                #self.last_obs = self.env.reset()

            last_stored_frame_idx = self.replay_buffer.store_frame(self.last_obs)

            #get observations to input to Q network (need to append prev frames)
            observations = self.replay_buffer.encode_recent_observation()

            #before learning starts, choose actions randomly
            if t < self.learning_starts:
                action = np.random.randint(self.num_actions)
            else:
                # epsilon greedy exploration
                sample = random.random()
                threshold = self.exploration.value(t)
                if sample > threshold:
                    obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                    with torch.no_grad():
                        #q_value_all_actions = self.Q(Variable(obs)).cpu()
                        ##q_value_all_actions = self.Q(Variable(obs))
                        ##action = ((q_value_all_actions).data.max(1)[1])[0]
                        action = self.Q(obs).max(1)[1].view(1,1)
                else:
                    action = torch.IntTensor([[np.random.randint(self.num_actions)]])[0][0]

            obs, reward, done = self.env.step(action)

            #clipping the reward, noted in nature paper
            reward = np.clip(reward, -1.0, 1.0)

            #store effect of action
            self.replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

            #reset env if reached episode boundary
            if done:
                print("----------Episode %d end!-------------" %len(self.env.episode_rewards))
                if self.env.canset():
                    obs = self.env.reset()
                else:
                    print("--------Episode number max-----------")
                    break

            #update last_obs
            self.last_obs = obs

            ### 3. Perform experience replay and train the network
            # if the replay buffer contains enough samples...
            if (t > self.learning_starts and
                    t % self.learning_freq == 0 and
                    self.replay_buffer.can_sample(self.batch_size)):
                #print("leaning.....,t:",t)

                # sample transition batch from replay memory
                # done_mask = 1 if next state is end of episode
                obs_t, act_t, rew_t, obs_tp1, done_mask = self.replay_buffer.sample(self.batch_size)
                obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
                act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
                rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
                obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
                done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)
                #print(rew_t.shape)

                # input batches to networks
                # get the Q values for current observations (Q(s,a, theta_i))
                q_values = self.Q(obs_t)
                q_s_a = q_values.gather(1, act_t.unsqueeze(1))
                q_s_a = q_s_a.squeeze()
                ##q_s_a = self.Q(obs_t).gather(1,act_t)

                if (self.double_dqn):
                    # ---------------
                    #   double DQN
                    # ---------------

                    # get the Q values for best actions in obs_tp1
                    # based off the current Q network
                    # max(Q(s', a', theta_i)) wrt a'
                    q_tp1_values = self.Q(obs_tp1).detach()
                    _, a_prime = q_tp1_values.max(1)

                    #get Q values from frozen network for next state and chosen action
                    q_target_tp1_values = self.Q_target(obs_tp1).detach()
                    q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                    q_target_s_a_prime = q_target_s_a_prime.squeeze()

                    # if current state is end of episode, then there if no next Q value
                    q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime
                    # old-method from pytorch2
                    #error = rew_t + self.gamma * q_target_s_a_prime - q_s_a
                    # new-method test
                    expected_state_action_values = rew_t + self.gamma * q_target_s_a_prime
                    loss = F.smooth_l1_loss(q_s_a,expected_state_action_values)

                else:
                    # -----------------
                    #   regular DQN
                    # -----------------

                    # get the Q values for best actions in obs_tp1
                    # based off frozen Q network
                    # max(Q(s', a', theta_i_frozen)) wrt a'
                    q_tp1_values = self.Q_target(obs_tp1).detach()
                    q_s_a_prime, a_prime = q_tp1_values.max(1)

                    # if current state is end of episode, then there is no next Q value
                    q_s_a_prime = (1 - done_mask) * q_s_a_prime

                    # Compute Bellman error
                    # r + gamma * Q(s', a', theta_i_frozen) - Q(s, a, theta_i)
                    # old-method
                    #error = rew_t + self.gamma * q_s_a_prime - q_s_a
                    # new method test
                    expected_state_action_values = rew_t + self.gamma * q_s_a_prime
                    loss = F.smooth_l1_loss(q_s_a,expected_state_action_values)

                # clip the error and flip--old
                #clipped_error = -1.0 * error.clamp(-1, 1)

                # backwards pass
                self.optimizer.zero_grad()
                #q_s_a.backward(clipped_error.data.unsqueeze(1))
                #q_s_a.backward(clipped_error)
                loss.backward()
                for param in self.Q.parameters():
                    param.grad.data.clamp(-1,1)

                # updata
                self.optimizer.step()
                self.num_param_updates += 1
                #print("num_param_updates:",self.num_param_updates)

                # update target Q nerwork weights with current Q network weights
                if self.num_param_updates % self.target_update_freq == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())

                # (2) Log values and gradients of the parameters (histogram)
                if t % self.Log_Every_N_Steps == 0:
                    for tag, value in self.Q.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), t+1)
                        self.logger.histo_summary(tag+'/grad', self.to_np(value.grad), t+1)
                ####

            ### 4. Log progress
            if t % self.Save_Model_Every_N_Steps == 0:
                if not os.path.exists("models"):
                    os.makedirs("models")
                add_str = ''
                if (self.double_dqn):
                    add_str = 'double'
                if (self.dueling_dqn):
                    add_str = 'dueling'
                model_save_path = "models/%s_%s_%d_%s.model" %(str("minecraft"), add_str, t, str(time.asctime()).replace(' ','_'))
                torch.save(self.Q.state_dict(), model_save_path)

            episode_rewards = self.env.get_average_rewards_per_episode()
            if len(episode_rewards) > 0:
                self.mean_episode_reward = np.mean(episode_rewards[-100:])
                self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
            if t % self.Log_Every_N_Steps == 0 and len(episode_rewards) > 0:
                print("---------------------------------------")
                print('Timestep %d' %(t,))
                print("learning started? %d" % (t >= self.learning_starts))
                print("num_param_updates:%d" % self.num_param_updates)
                print("mean reward (100 episodes) %f" % self.mean_episode_reward)
                print("best mean reward %f" % self.best_mean_episode_reward)
                print("episodes_done %d"  % len(episode_rewards))
                print("exploration %f" % self.exploration.value(t))
                print("learning_rate %f" % self.optimizer_spec.kwargs['lr'])
                #sys.stdout,flush()

                #=================== TensorBoard logging =============#
                # (1) Log the scalar values
                info = {
                    'learning_started': (t > self.learning_starts),
                    'num_episodes': len(episode_rewards),
                    'exploration':self.exploration.value(t),
                    'learning_rate': self.optimizer_spec.kwargs['lr'],
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, t+1)

                if len(episode_rewards) > 0:
                    info = {
                        'last_episode_rewards': episode_rewards[-1],
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, t+1)

                if (self.best_mean_episode_reward != -float('inf')):
                    info = {
                        'mean_episode_reward_last_100': self.mean_episode_reward,
                        'best_mean_episode_reward': self.best_mean_episode_reward
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, t+1)

            if t % self.Save_Model_Every_N_Steps == 0:
                episode_records = np.array(episode_rewards)
                exploration = np.array(self.exploration.value(t))

                if not os.path.exists("perform_records"):
                    os.makedirs("perform_records")
                add_str = ''
                if (self.double_dqn):
                    add_str = 'double'
                if (self.dueling_dqn):
                    add_str = 'dueling'
                save_path = "perform_records/%s_%s_ep%d.npy" %(str("episode_records"), add_str, t)
                save_path_2 = "perform_records/%s_%s_ep%d.npy" %(str("exploration"), add_str, t)

                np.save(save_path, episode_records)
                np.save(save_path_2, exploration)
