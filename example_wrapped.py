"""Example of interacting with the rllib environment. Runs episodes with random legal actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

import numpy as np
from absl import logging
import time
import pprint

from masc2.rllib.env import RLlibStarCraft2Env

logging.set_verbosity(logging.DEBUG)

def main():
    distribution_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.6, 0.3, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    env = RLlibStarCraft2Env(
        env_config={"map_name": "10gen_terran", "capability_config": distribution_config}
    )

    env_info = env._env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 2

    print("Training episodes")
    for e in range(n_episodes):
        obs, _ = env.reset()
        print('Episode:', e)
        done = False
        episode_reward = 0

        while not done:
            actions = {}
            for agent_id in obs:
                avail_actions = obs[agent_id]['action_mask']
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions[agent_id] = action

            obs, reward, terminated, _, info = env.step(actions)
            time.sleep(0.15)
            done=terminated['__all__']
            episode_reward += sum(reward.values())
        print(f'Info at end of episode {e}:')
        pprint.pprint(info)
        print("Total reward in episode {} = {}".format(e, episode_reward))
    # env._env.save_replay() # Uncomment this to save a replay

if __name__ == "__main__":
    main()