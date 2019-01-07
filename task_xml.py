video_width  = 200
video_height = 160

apple_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawItem x="46" y="5" z="47" type="apple"/>

                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="10"/>
                    <Item type="wool" reward="100"/>
                  </RewardForCollectingItem>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


find_stone_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawBlock type="stone" x="48" y="5" z="48"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                        <min x="1" y="0" z="1"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="100"/>
                    <Item type="wool" reward="100"/>
                  </RewardForCollectingItem>
                  <RewardForDamagingEntity>
                    <Mob type="Sheep" reward="500"/>
                  </RewardForDamagingEntity>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

find_sheep_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawEntity x="48" y="5" z="48" type="Sheep"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                        <min x="1" y="0" z="1"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ObservationFromNearbyEntities>
                    <Range name="entity" xrange="1" yrange="0" zrange="1"/>
                  </ObservationFromNearbyEntities>
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="100"/>
                    <Item type="wool" reward="100"/>
                  </RewardForCollectingItem>
                  <RewardForDamagingEntity>
                    <Mob type="Sheep" reward="500"/>
                  </RewardForDamagingEntity>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

tools_stone_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawBlock type="stone" x="45" y="5" z="45"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                        <min x="1" y="0" z="1"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="100"/>
                    <Item type="wool" reward="100"/>
                    <Item type="cobblestone" reward="100"/>
                  </RewardForCollectingItem>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

tools_sheep_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawEntity x="45" y="5" z="45" type="Sheep"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="60000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                        <min x="1" y="0" z="1"/>
                        <max x="3" y="0" z="3"/>
                    </Grid>
                  </ObservationFromGrid>
                  <ObservationFromNearbyEntities>
                    <Range name="entity" xrange="1" yrange="0" zrange="1"/>
                  </ObservationFromNearbyEntities>
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="100"/>
                    <Item type="wool" reward="100"/>
                    <Item type="cobblestone" reward="100"/>
                  </RewardForCollectingItem>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="60000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

hdrln_missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    <DrawCuboid type="air" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>
                    <DrawCuboid type="grass" x1="0" y1="2" z1="0" x2="100" y2="4" z2="100"/>

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

                    <DrawBlock type="fence_gate" x="49" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="48" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="47" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="46" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="45" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="44" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="43" y="5" z="41"/>
                    <DrawBlock type="fence_gate" x="42" y="5" z="41"/>


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

                    <DrawBlock type="stone" x="48" y="5" z="48"/>
                    <DrawEntity x="43" y="5" z="47" type="Sheep"/>
                    <DrawItem x="45" y="5" z="48" type="apple"/>


                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>

              <AgentSection mode="Survival">
                <Name>Gilgamesh</Name>
                <AgentStart>
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
                  <ObservationFromRay/>
                  <ObservationFromFullInventory/>
                  <RewardForCollectingItem>
                    <Item type="apple" reward="50"/>
                    <Item type="wool" reward="50"/>
                    <Item type="cobblestone" reward="100"/>
                  </RewardForCollectingItem>
                  <ContinuousMovementCommands turnSpeedDegs="150"/>
                  <InventoryCommands/>
                  <MissionQuitCommands/>
                  <AgentQuitFromTimeUp timeLimitMs="300000"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
