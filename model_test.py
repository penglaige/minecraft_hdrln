from __future__ import print_function
from __future__ import division
# Start the training environment
# Find, break, use tools, hdrln
# Find : item-apple, tool-axe,shears
# Break: item-block, tool-axe,shears
# Use:   item-sheep, tool-axe,shears
# hdrln: item-apple, block, sheep, tool-axe,shears
import torch
import torch.optim as optim
import argparse

from task_xml import apple_missionXML
from model import DQN, Dueling_DQN
from utils.schedules import *
from utils.minecraft_wrappers import ENV

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import json
import random
import logging
import struct
import socket
import malmoutils

# Global Variables
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
#TARGET_UPDATE_FREQ = 1000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
LEARNING_STARTS = 50000
#for testing
#LEARNING_STARTS = 1000

RESIZE_MODE   = 'scale'
RESIZE_WIDTH  = 84
RESIZE_HEIGHT = 84

num_actions = 3

def stopping_criterion(agent_host, t):
    world_state = agent_host.getWorldState()
    return not world_state.is_mission_running

# ---------------read models
model_save_path = "models/test.model"
model = torch.load(model_save_path)

# --------------------------

#----------- Test setting -----------------------------------------------------------------
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import numpy as np
from collections import namedtuple
from utils.replay_buffer import *
from logger import Logger

#CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# dqn_learner
class DQNAgent(object):
    """A dqn learner agent."""

    def __init__(self,env,q_func,num_actions,
        exploration=LinearSchedule(1000000, 0.1),
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

        #create replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len,self.img_w,self.img_h,"scale")

        ###### RUN SETTING ####
        self.t = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = None
        #########################

    #Set the logger
    def to_np(self,x):
        return x.data.cpu().numpy() #???

    def run(self, agent_host):
        #run many times
        if self.last_obs is None:
            self.last_obs = self.env.reset()

        self.Q.load_state_dict(model)
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

            # epsilon greedy exploration
            sample = random.random()
            #threshold = self.exploration.value(t)
            threshold = 0.6
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

            #update last_obs
            self.last_obs = obs


#----------- Test setting -----------------------------------------------------------------

def stopping_criterion(agent_host, t):
    world_state = agent_host.getWorldState()
    return not world_state.is_mission_running

malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
train, gpu, double_dqn, dueling_dqn = malmoutils.get_options(agent_host)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

missionXML = apple_missionXML

validate = True
my_mission = MalmoPython.MissionSpec( missionXML, validate )

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.LATEST_REWARD_ONLY)

if not train:
    num_reps = 5
else:
    num_reps = 400

print("num_reps:",num_reps)

my_mission_record = MalmoPython.MissionRecordSpec()
if recordingsDirectory:
    my_mission_record.recordRewards()
    my_mission_record.recordObservations()
    my_mission_record.recordCommands()
    if agent_host.receivedArgument("record_video"):
        my_mission_record.recordMP4(24,2000000)

env = ENV(agent_host, my_mission, my_mission_record, logger, recordingsDirectory, MAX_EPISODE=num_reps)

# ------------Command Parser-------------------------
if (gpu != None):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("CUDA Device: %d" %torch.cuda.current_device())

seed = 0
#--------------------------- Agent setting ------------------------------------------------------------
if dueling_dqn:
    agent = DQNAgent(
                env=env,
                q_func=Dueling_DQN,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=1,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
else:
    agent = DQNAgent(
                env=env,
                q_func=DQN,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=1,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
#--------------------------- Begin Minecraft game -----------------------------------------------------
agent.run(agent_host)
#--------------------------- Begin Minecraft game -----------------------------------------------------
print("-----------------------Training ends-----------------------")
