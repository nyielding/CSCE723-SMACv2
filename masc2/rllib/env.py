from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from typing import Optional

from gymnasium.spaces import Discrete, Box, Dict

from ray import rllib
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


class RLlibStarCraft2Env(rllib.MultiAgentEnv):
    """Wraps a smac StarCraft env to be compatible with RLlib multi-agent."""

    def __init__(self, env_config={}, **smac_args):
        super().__init__()
        capability_default = {
            # Optionally write a default capability config here
        }

        # Pull config from env_config
        smac_args['map_name'] = env_config.get("map_name", '10gen_terran')
        smac_args['capability_config'] = env_config.get("capability_config", capability_default)
        self.shape_reward = env_config.get('shape_reward', False)
        self.save_replays = env_config.get("save_replays", False)
        self.replay_duration = env_config.get("replay_duration", None)
        smac_args['replay_dir'] = env_config.get("replay_folder", "replays")
        
        # Initialize the SMAC2 environment
        self._env = StarCraftCapabilityEnvWrapper(**smac_args)
        self.env_info = self._env.get_env_info()
        self.n_agents = self.env_info["n_agents"]

        # Must reset env here or else self._init_agents() will not read correct unit ids
        self._env.reset()

        self._agent_ids = self._init_agents()
        self._ready_agents = self._agent_ids[:]

        self.observation_space = Dict(
            {
                "observations": Box(
                    -1, 
                    1, 
                    shape=(self._env.get_obs_size(),), 
                    dtype=np.float32
                ),
                "action_mask": Box(
                    0, 
                    1, 
                    shape=(self._env.get_total_actions(),), 
                    dtype=np.int8
                ),
            }
        )
        self.action_space = Discrete(self._env.get_total_actions())

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        # print("RESET SCII ENV...")
        self._step = 0
        return_obs, infos = {}, {}
        obs_list, _ = self._env.reset()
        for i, obs in enumerate(obs_list):
            agent_id = self._get_agent_id(i)
            return_obs[agent_id] = {
                "observations": np.array(obs, dtype=np.float32),
                "action_mask": np.array(self._env.get_avail_agent_actions(i), dtype=np.int8),
            }
            infos[agent_id] = self._env.get_env_info()

        self._ready_agents = list(return_obs.keys())
        # print("RESET COMPLETE")
        return return_obs, infos

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            terms (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            truns (dict): unused, always set all to false
            infos (dict): Optional info values for each agent id.
        """
        self._step += 1
        # print('action_dict: ', action_dict)
        actions = self._convert_actions(action_dict)

        if len(actions) != len(self._ready_agents):
            raise ValueError(
                "Unexpected number of actions: {}".format(
                    action_dict,
                )
            )
        # print("STEPPING SCII ENV...")
        reward, term, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        # print('obs_list length: ', len(obs_list))
        return_obs, rewards, terms, truns, infos = {}, {}, {}, {}, {}
        for i, obs in enumerate(obs_list):
            agent_id = self._get_agent_id(i)
            return_obs[agent_id] = {
                "observations": np.array(obs, dtype=np.float32),
                "action_mask": np.array(self._env.get_avail_agent_actions(i), dtype=np.int8),
            }
            infos[agent_id] = info
            terms[agent_id] = term
            rewards[agent_id] = self._reward_shape(agent_id=agent_id, action_dict=action_dict, obs=obs, reward_in=reward, agents_alive=len(obs_list))
            truns[agent_id] = False
        terms["__all__"] = term
        truns["__all__"] = False

        self._ready_agents = list(range(len(obs_list)))
        # print("STEP COMPLETE")
        if terms["__all__"] and self.save_replays:
            print("Battle won?: ", info.get('battle_won', 0))
            if self.replay_duration:
                print("Episode Count: ", self._env.env._episode_count)
                if self._env.env._episode_count == self.replay_duration:
                    self._env.save_replay()
            else:
                self._env.save_replay()

        return return_obs, rewards, terms, truns, infos

    def render(self) -> None:
        pass

    def close(self):
        """Close the environment"""
        self._env.close()

    def _init_agents(self):
        """Initializes agents to be indexed with type names for multi-policy mapping"""
        last_type = ""
        agents_ids = []
        self.agent_map_num = {}
        self.agent_map_id = {}
        i = 0
        for agent_num, agent_info in self._env.env.agents.items():
            if agent_info.unit_type == self._env.env.marine_id:
                agent_type = "marine"
            elif agent_info.unit_type == self._env.env.marauder_id:
                agent_type = "marauder"
            elif agent_info.unit_type == self._env.env.medivac_id:
                agent_type = "medivac"
            elif agent_info.unit_type == self._env.env.hydralisk_id:
                agent_type = "hydralisk"
            elif agent_info.unit_type == self._env.env.zergling_id:
                agent_type = "zergling"
            elif agent_info.unit_type == self._env.env.baneling_id:
                agent_type = "baneling"
            elif agent_info.unit_type == self._env.env.stalker_id:
                agent_type = "stalker"
            elif agent_info.unit_type == self._env.env.colossus_id:
                agent_type = "colossus"
            elif agent_info.unit_type == self._env.env.zealot_id:
                agent_type = "zealot"
            else:
                raise AssertionError(f"agent type {agent_type} not supported")

            if agent_type == last_type:
                i += 1
            else:
                i = 0

            agents_ids.append(f"{agent_type}_{i}")
            self.agent_map_id[agents_ids[-1]] = agent_num
            self.agent_map_num[agent_num] = agents_ids[-1]
            last_type = agent_type

        return agents_ids

    def _get_agent_id(self, num):
        """Get the string name for agent ids in this env for policy mapping, from smacv2 env agent index"""
        return self.agent_map_num[num]
    
    def _get_agent_num(self, name):
        """Get the number that the starcraft smacv2 base environment uses for agent ids"""
        return self.agent_map_id[name]
    
    def _convert_actions(self, action_dict):
        action_list = [0]*self._env.env.n_agents
        for agent_id in self._agent_ids:
            agent_num = self._get_agent_num(agent_id)
            if agent_id in action_dict:
                action_list[agent_num] = action_dict[agent_id]
        return action_list
    
    def _reward_shape(self, agent_id, action_dict, obs, reward_in, agents_alive):
        """
        Improve the reward shaping here.

        By default, the agent gets 1/N of the step reward, where N is the number of agents returning obs.

        Give an additional reward if the agent chose to shoot the lowest health enemy it can see, or heal the lowest ally it can see
        """
        reward_out = reward_in/agents_alive
        if self.shape_reward:
            if agent_id.split('_')[0] == 'medivac':
                pass # FIXME: Implement reward shaping for medivac
                # HINTS: 
                # Check to see if the action was to heal an ally
                # Check to see if the ally healed was the lowest health ally
                # # Get ally health from the observations (since the agent can only see allies in range)
                # # From obs_feature_names get the index of the ally healths
                # # Get the ally healths from obs that are greater than 0 (0 is dead or out of range)
                # # Get the lowest health ally, which could be multiple if they have the same health
                # # If the action was to heal the lowest health ally, give a small reward, say 0.25 in addition to normal reward

            else:
                pass # FIXME: implement reward shaping for other agents
                # HINTS: This should be similar to the medivac reward shaping,
                # Except the focus should be on enemy_health instead of ally_health

        return reward_out
