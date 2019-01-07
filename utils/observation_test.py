# Start the training environment
# Find, break, use tools, hdrln
# Find : item-apple, tool-axe,shears
# Break: item-block, tool-axe,shears
# Use:   item-sheep, tool-axe,shears
# hdrln: item-apple, block, sheep, tool-axe,shears
from __future__ import print_function
from __future__ import division

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

malmoutils.fix_print()


video_width = 200
video_height = 160

find_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

              <About>
                <Summary>HDRLN</Summary>
              </About>

              <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;1*minecraft:bedrock,3*minecraft:dirt,1*minecraft:grass;2;village"/>
                  <DrawingDecorator>
                    <DrawBlock type="fence_gate" x="50" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="49" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="50"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="50"/>

                    <DrawBlock type="fence_gate" x="50" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="41"/>

                    <DrawBlock type="fence_gate" x="50" y="5" z="49" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="48" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="47" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="46" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="45" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="44" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="43" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="42" face="WEST"/>
                    <DrawBlock type="fence_gate" x="50" y="5" z="41" face="WEST"/>

                    <DrawBlock type="fence_gate" x="41" y="5" z="49" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="48" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="47" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="46" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="45" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="44" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="43" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="42" face="WEST"/>
                    <DrawBlock type="fence_gate" x="41" y="5" z="41" face="WEST"/>

                    <DrawItem x="45" y="5" z="45" type="apple"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
                    <Placement x="42" y="5" z="45" yaw="-90" pitch="30"/>
                    <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                        <InventoryItem slot="7" type="shears"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <VideoProducer>
                    <Width>''' + str(video_width) + '''</Width>
                    <Height>''' + str(video_height) + '''</Height>
                  </VideoProducer>
                  <ObservationFromGrid>
                    <Grid name="front3x3">
                        <min x="0" y="0" z="0"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ContinuousMovementCommands turnSpeedDegs="300"/>
                  <InventoryCommands/>
                  <AgentQuitFromCollectingItem>
                    <Item type="apple"/>
                  </AgentQuitFromCollectingItem>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

#Create default Malmo objects:

agent_host = MalmoPython.AgentHost()

try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(find_missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ",)
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print
print("Mission running ",)
# Loop until mission ends:
#agent_host.sendCommand("turn 1")
#time.sleep(0.1)
#agent_host.sendCommand("turn 0")
agent_host.sendCommand("move 0.1")

while world_state.is_mission_running:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    if world_state.number_of_observations_since_last_state > 0:
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'front3x3', 0)



print()
print("Mission ended")
# Mission has ended.
