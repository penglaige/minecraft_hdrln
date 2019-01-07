from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
from collections import deque

from builtins import range
from past.utils import old_div
import MalmoPython
import random
import time
import logging
import struct
import socket
import os
import sys
import malmoutils
import json

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ENV():
    def __init__(self, agent_host, my_mission, my_mission_record, logger,recordingsDirectory, MAX_EPISODE):
        self.agent_host = agent_host
        self.my_mission = my_mission
        self.my_mission_record = my_mission_record
        self.logger = logger
        self.cumulative_rewards = 0.0
        self.episode_time = 300.0
        self.count = 0
        self.action = 0
        #self.cumulative_rewards_from_agent = 0.0
        self.episode_start_time = None
        self.episode_rewards = []
        self.episode_time_average_rewards = []
        self.max_episode = MAX_EPISODE
        self.recordingsDirectory=recordingsDirectory
        self.MAX_EPISODE = MAX_EPISODE
        self.width = None
        self.height = None
        self.channels = None
        self.apple_num = 0
        #-------random--------
        #self.apple_space = self.get_random_space(0)
        #self.agent_space = self.get_random_space(1)
        #self.apple_x,self.apple_z,self.agent_x,self.agent_z,self.agent_yaw = self.get_random_position()
        #-------fix-------
        #_,_,_,_,agent_yaw = self.get_random_position()
        #self.my_mission.drawItem( 48,5,48,"apple")
        #self.my_mission.startAtWithPitchAndYaw( 43.5,5,43.5,30,agent_yaw)

    def get_random_position(self,item):
        stone = [(45.5,44.5,0), (46,44.5,45), (45,44.5,315)]
        sheep = [(45,44,0),(46,45,90),(45,46,180),(44,45,270),(46,44,45), (46,46,135),(44,46,225),(44,44,315)]
        if item == "stone":
            x,z,yaw = random.choice(stone)
        else:
            x,z,yaw = random.choice(sheep)
        return x,z,yaw

    def canset(self):
        #print("canset",len(self.episode_rewards),self.MAX_EPISODE)
        if len(self.episode_rewards) < self.MAX_EPISODE:
            return 1
        else:
            return 0

    def reset(self):
        #  ------ run mission ---------
        if self.recordingsDirectory:
            self.my_mission_record.setDestination(self.recordingsDirectory + "//" + "Mission_" + str(len(self.episode_rewards) + 1) + ".tgz")

        agent_yaw = random.choice(range(0,360))
        # 0 stont; 1 sheep
        stone_or_sheep = random.choice(range(2))
        if stone_or_sheep:
            #sheep
            print("sheep")
            mission = self.my_mission[1]
            x,z,yaw = self.get_random_position("sheep")
            mission.startAtWithPitchAndYaw( x,5,z,30,yaw)
        else:
            print("stone")
            mission = self.my_mission[0]
            x,z,yaw = self.get_random_position("stone")
            mission.startAtWithPitchAndYaw( x,5,z,30,yaw)
        #mission = self.my_mission

        #mission.drawItem( 48,5,48,"apple")

        #mission.startAtWithPitchAndYaw( 44,5,44,30,270)
        mission.forceWorldReset()

        time.sleep(0.5)
        self.agent_host.sendCommand("quit")
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( mission, self.my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    self.logger.error("Error starting mission: %s" % e)
                    exit(1)
                else:
                    time.sleep(2)

        self.logger.info('Mission %s', len(self.episode_rewards)+1)
        self.logger.info("Waiting for the mission to start")
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        # game start
        self.agent_host.sendCommand("hotbar.5 1")
        self.episode_time = 300.0
        self.count = 0
        self.episode_start_time = time.time()
        self.cumulative_rewards = 0.0
        self.apple_num = 0
        mission = None

        frame = self.get_frame()
        #self.agent_host.sendCommand( "move 1" )
        #self.agent_host.sendCommand( "attack 1")
        return frame


    def get_frame(self,ALL=False):
        world_state = self.agent_host.getWorldState()
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            #self.logger.info("Waiting for frames...")
            time.sleep(0.05)
            world_state = self.agent_host.getWorldState()


        if len(world_state.video_frames) > 0:
            #self.logger.info("Got frame!")
            if not self.width or not self.height or not self.channels:
                self.width = world_state.video_frames[0].width
                self.height = world_state.video_frames[0].height
                self.channels = world_state.video_frames[0].channels
            frame = world_state.video_frames[0].pixels #frame h w C
        elif not world_state.is_mission_running:
            self.logger.info("Episode end!")
            frame = np.zeros((self.height,self.width,self.channels))

        frame = np.array(frame).reshape(self.height,self.width,self.channels)

        reward = self.get_reward(world_state)
        #self.cumulative_rewards += reward_from_obs
        self.cumulative_rewards += reward
        #self.cumulative_rewards_from_agent += reward_from_agent
        #print("new_cumulative_rewards: %f" % self.cumulative_rewards)

        done = self.done(world_state)

        if done:
            print("cumulative_counts: %d" % self.cumulative_rewards)
            self.episode_rewards.append(self.cumulative_rewards)
            print("count: %d" %self.count)
            print("average_rewards: %f" % (self.cumulative_rewards / self.count))
            self.episode_time_average_rewards.append(self.cumulative_rewards / self.count)
            self.cumulative_rewards = 0.0
            self.count = 0

        if not ALL:
            return frame
        else:
            return frame, reward, done

    def done(self,world_state):
        done_from_time = not world_state.is_mission_running
        done_from_task = False

        if self.count >= 1:
            episode_end_time = time.time()
            self.episode_time = episode_end_time - self.episode_start_time
            done_from_task = True
            if world_state.is_mission_running:
                self.agent_host.sendCommand(" move 0")
                for i in range(1):
                    #self.agent_host.sendCommand("quit")
                    print("try to quit mission...")
                    time.sleep(0.04)
            #if self.cumulative_rewards != 500.0:
                #self.cumulative_rewards = 500.0
        return done_from_time or done_from_task



    def get_reward(self,world_state):
        reward = 0

        if len(world_state.observations) < 1:
            reward_from_inventory = 0
        else:
            msg = world_state.observations[-1].text
            information = json.loads(msg)
            LineOfSight = information.get(u'LineOfSight', 0)
            item_num = information.get(u'InventorySlot_0_size', 0)
            if item_num:
                reward_from_inventory = 100.0
            else:
                reward_from_inventory = 0.0

        if len(world_state.rewards) < 1:
            reward = max(reward, reward_from_inventory)
        else:
            #reward = world_state.rewards[-1].getValue()
            reward = 100.0

        #if reward_from_stone == 50 or reward_from_sheep == 50:
            #print("rewards:",reward_from_stone,reward_from_sheep)


        return reward


    def step(self, action):
        self.action = action
        print("action:",action)
        self.count += 1
        if action == 0:
            # axes
            self.agent_host.sendCommand("hotbar.9 1")
            self.agent_host.sendCommand("attack 1")
            time.sleep(0.5)
            self.agent_host.sendCommand("attack 0")
            time.sleep(5)
            self.agent_host.sendCommand("move 1")
            time.sleep(0.7)
            self.agent_host.sendCommand("move 0")
            time.sleep(2)
        elif action == 1:
            #shears
            self.agent_host.sendCommand("hotbar.8 1")
            self.agent_host.sendCommand("use 1")
            self.agent_host.sendCommand("use 0")
            time.sleep(5)
            self.agent_host.sendCommand("move 1")
            time.sleep(0.7)
            self.agent_host.sendCommand("move 0")
            time.sleep(2)

        new_obs, reward, done = self.get_frame(ALL=True)
        return new_obs, reward, done

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_average_rewards_per_episode(self):
        return self.episode_time_average_rewards
