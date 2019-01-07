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

from task_xml import hdrln_missionXML
from model import DQN, Dueling_DQN
from hdrln_learn_wo_apple import DQNAgent, OptimizerSpec
from utils.schedules import *
from utils.hdrln_wo_minecraft_wrappers import ENV

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
TARGET_UPDATE_FREQ = 8000
#TARGET_UPDATE_FREQ = 1000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = hdrln_LinearSchedule(40000,250000, 0.1)
LEARNING_STARTS = 60000
#for testing
#LEARNING_STARTS = 1000

RESIZE_MODE   = 'scale'
RESIZE_WIDTH  = 84
RESIZE_HEIGHT = 84

num_actions = 5

def stopping_criterion(agent_host, t):
    world_state = agent_host.getWorldState()
    return not world_state.is_mission_running

optimizer = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
)

# ---------------read models
model_save_path = "skill_model/"
apple_agent = torch.load(model_save_path + "apple_250ep.model")
find_agent = torch.load(model_save_path + "find7_49wt.model")
tool_agent = torch.load(model_save_path + "tools.model")

skill_pool = [find_agent, tool_agent]
skill_space = [4,2]
skill_teminal_step = [40,1]

# --------------------------

#----------- Minecraft environment setting -----------------------------------------------------------------
malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
train, gpu, double_dqn, dueling_dqn = malmoutils.get_options(agent_host)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

missionXML = hdrln_missionXML


validate = True
my_mission = MalmoPython.MissionSpec( missionXML, validate )

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS )

if not train:
    num_reps = 5
else:
    num_reps = 1000

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
                optimizer_spec=optimizer,
                num_actions=num_actions,
                skill_pool=skill_pool,
                skill_space=skill_space,
                skill_teminal_step=skill_teminal_step,
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
                optimizer_spec=optimizer,
                num_actions=num_actions,
                skill_pool=skill_pool,
                skill_space=skill_space,
                skill_teminal_step=skill_teminal_step,
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
