# MASC2 Arena
Project for training agents in the Multi-agent StarcraftII sandbox, SMACv2 (Starcraft Multi-Agent Challenge v2)

Environment project repo: https://github.com/oxwhirl/smacv2

V1 repo for additional information: https://github.com/oxwhirl/smac/blob/master/README.md

Use PyMARL which is setup for this? (it is like sb3 or rllib) https://github.com/oxwhirl/pymarl/tree/master

# Replays
## Saving a replay 
Call the method save_replay method on the environment at the end of an episode, `env.save_replay()`. 

## Watching a replay
1. On a Windows or Mac machine, download battle.net (make a free account if you don't have one) 
2. Install any edition of StarCraftII (SC2) -- the "Starter Edition" is free.
3. Download the maps from 
```
https://github.com/oxwhirl/smacv2/releases/tag/maps
```
4. Find your SC2 installation folder on machine, then create a directory inside it for maps: `.../StarCraftII/Maps/SMAC_Maps/`
5. Unzip the contents of the SMACv2 Maps from `SMAC_Maps.zip` into `.../StarCraftII/Maps/SMAC_Maps/`
6. Download replay file onto this machine, double click replay file. Sign into your battle.net account in SC2 if needed.

# Where information is located
The original SMAC docs are still mostly relevant https://github.com/oxwhirl/smac/blob/master/docs/smac.md 
## Information on how the units and initial position are decided
class `StarCraftCapabilityEnvWrapper` in smacv2>env>starcraft2>wrapper.py

## Information on observation and actions of agent/unit
class `StarCraft2Env` in smacv2>env>starcraft2>starcraft2.py

# Action Space
Action is a discrete choice that has N options based on M enemies:

Noop (only available if dead, and only option if dead):
```
    0
```
Stop:
```
    1
```

Discrete cardinal direction movements according to:

```
    NORTH = 2
    SOUTH = 3
    EAST = 4
    WEST = 5
```
Attack enemy number (or heal ally number if medivac):
```
    attack enemy 1 = 6
    attack enemy 2 = 7
    ...
    attack enemy M = 5+M
```

### 8 v 8 example action space
For 8v8 the `action_space` for each agent will be Discrete(14) with the options for an action:
```
0: noop
1: stop
2: move north
3: move south
4: move east
5: move west
6: attack enemy 1
7: attack enemy 2
8: attack enemy 3
9: attack enemy 4
10: attack enemy 5
11: attack enemy 6
12: attack enemy 7
13: attack enemy 8
```
These actions are MASKED. An action may only be chosen if the corresponding index of the action in the `action_mask == 1`. If an action x is chosen where `action_mask[x] == 0` then the environment will return an error. The action mask is included in the observations

# Observation Space
The observation space contains a list of values that include the following in order for EACH of N agents. The observations include both the action masking and the observation itself.

The total observation for the step output will be a dictionary with the agent_id as the key ie
```
{
    'marine_0': {'action_mask': array[...], 'observations': ob1}, 
    'marine_1': {'action_mask': array[...], 'observations': ob2}, 
    'marauder_0': {'action_mask': array[...], 'observations': ob3}, 
    ..., 
    'marauder_N': {'action_mask': array[...], 'observations': obN}
}
```

### Movement options
Action mask for movement in N,S,E,W directions is also included as the first 4 values in 'observations' 
```
['move_action_north',
 'move_action_south',
 'move_action_east',
 'move_action_west']
```

I.e. if movement possible in all direction but north:
```
[0,1,1,1]
```

### Enemy info
The following 5 values for each enemy. If unknown due to observability, value is 0:
```
['enemy_shootable_0',
 'enemy_distance_0',
 'enemy_relative_x_0',
 'enemy_relative_y_0',
 'enemy_health_0',
 'enemy_unit_type_0_bit_0',
 'enemy_unit_type_0_bit_1',
 'enemy_unit_type_0_bit_2',]
```
I.e. if there are 2 enemies and the first is 3 away in x dir, 4 away in y dir, and has 75% health, while the second is unobservable:
```
[1,5,3,4,0.75,0,0,0,0,0]
```
### Ally info
The following 5 values for each ally that is observable:
```
['ally_visible_1',
 'ally_distance_1',
 'ally_relative_x_1',
 'ally_relative_y_1',
 'ally_health_1',
 'ally_unit_type_1_bit_0',
 'ally_unit_type_1_bit_1',
 'ally_unit_type_1_bit_2',]
```
 ### Own health
 A single value for own health:

 ```
 [1 ]

 ```

### Full feature labels:
Stored in the unwrapped environment under `obs_feature_names`

This example is for 5 vs 5 units

```
['move_action_north',
 'move_action_south',
 'move_action_east',
 'move_action_west',
 'enemy_shootable_0',
 'enemy_distance_0',
 'enemy_relative_x_0',
 'enemy_relative_y_0',
 'enemy_health_0',
 'enemy_unit_type_0_bit_0',
 'enemy_unit_type_0_bit_1',
 'enemy_unit_type_0_bit_2',
 'enemy_shootable_1',
 'enemy_distance_1',
 'enemy_relative_x_1',
 'enemy_relative_y_1',
 'enemy_health_1',
 'enemy_unit_type_1_bit_0',
 'enemy_unit_type_1_bit_1',
 'enemy_unit_type_1_bit_2',
 'enemy_shootable_2',
 'enemy_distance_2',
 'enemy_relative_x_2',
 'enemy_relative_y_2',
 'enemy_health_2',
 'enemy_unit_type_2_bit_0',
 'enemy_unit_type_2_bit_1',
 'enemy_unit_type_2_bit_2',
 'enemy_shootable_3',
 'enemy_distance_3',
 'enemy_relative_x_3',
 'enemy_relative_y_3',
 'enemy_health_3',
 'enemy_unit_type_3_bit_0',
 'enemy_unit_type_3_bit_1',
 'enemy_unit_type_3_bit_2',
 'enemy_shootable_4',
 'enemy_distance_4',
 'enemy_relative_x_4',
 'enemy_relative_y_4',
 'enemy_health_4',
 'enemy_unit_type_4_bit_0',
 'enemy_unit_type_4_bit_1',
 'enemy_unit_type_4_bit_2',
 'ally_visible_1',
 'ally_distance_1',
 'ally_relative_x_1',
 'ally_relative_y_1',
 'ally_health_1',
 'ally_unit_type_1_bit_0',
 'ally_unit_type_1_bit_1',
 'ally_unit_type_1_bit_2',
 'ally_visible_2',
 'ally_distance_2',
 'ally_relative_x_2',
 'ally_relative_y_2',
 'ally_health_2',
 'ally_unit_type_2_bit_0',
 'ally_unit_type_2_bit_1',
 'ally_unit_type_2_bit_2',
 'ally_visible_3',
 'ally_distance_3',
 'ally_relative_x_3',
 'ally_relative_y_3',
 'ally_health_3',
 'ally_unit_type_3_bit_0',
 'ally_unit_type_3_bit_1',
 'ally_unit_type_3_bit_2',
 'ally_visible_4',
 'ally_distance_4',
 'ally_relative_x_4',
 'ally_relative_y_4',
 'ally_health_4',
 'ally_unit_type_4_bit_0',
 'ally_unit_type_4_bit_1',
 'ally_unit_type_4_bit_2',
 'own_health',
 'own_unit_type_bit_0',
 'own_unit_type_bit_1',
 'own_unit_type_bit_2']
 ```

# Bugs
- "render" attempts to return a pygame or rgb_array do not work due to internal bugs

# Student assignment TODOs
1. Run and play around with both the `example.py` and `example_wrapped.py` to better understand the environment 
2. Write the callback to record the win rate in the custom metrics.
3. Write the training script `train.py` in order to train policies.
4. Train policies for the units marine, marauder, and medivac (Terran race).
5. Write the reward shaping method and enable it, then train a new set of policies.
6. Copy a checkpoint and replay file to the `submission/` folder, complete the questions in `SUBMISSION.md`.

## Submission folder info
- There is a pytest to check the structure of your submission folder in `tests/`.

Once you have your trained policy's best checkpoint folder, copy this folder to `./submission/YOUR-LAST-NAME`. Only copy ONE checkpoint, not the whole results folder. A checkpoint is a **folder**, not an individual *file*, and looks roughly like: 
```
/home/developer/CSCE723-SMACV2/ray_results/SicMicroBro/PPO_TvT_bb575_00000_0_2024-02-21_03-24-33/checkpoint_000004
```
The directory of this checkpoint should contain these files:
```
checkpoint_000004/
|-- policies/
|   |-- marauder/
|   |   |-- policy_state.pkl
|   |   |-- rllib_checkpoint.json
|   |-- marine/
|   |   |-- policy_state.pkl
|   |   |-- rllib_checkpoint.json
|   |-- medivac/
|   |   |-- policy_state.pkl
|   |   |-- rllib_checkpoint.json
|
|-- algorithm_state.pkl
|-- rllib_checkpoint.json
```
