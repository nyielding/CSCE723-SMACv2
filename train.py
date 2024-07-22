"""
A training script using tune.Tuner to handle a complex experiment. Uses a specified Gymnasium env and an Algorithm from RLlib
"""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import AlgorithmConfig
import ray
import pathlib
import numpy as np
from ray import tune, train, air
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from masc2.rllib.env import RLlibStarCraft2Env
from masc2.rllib.callbacks import WinRatioCallback
from ray.rllib.examples._old_api_stack.models.action_mask_model import ActionMaskModel, TorchActionMaskModel
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

local_mode = False
use_wandb = False # If True, remember to `wandb login` in terminal before running
training_iterations = 500 # max iterations before stopping
num_cpu = 10 # Set to your CPU count
num_gpus = 0
num_eval_workers = 1 # Reserve one environment for evaluating periodically

driver_cpu = 1 # Reserve one cpu for the driver
# How CPUs are spread
num_env_runners = num_cpu - driver_cpu - num_eval_workers

# Folder to where the results and policies are to be saved
ray_results = pathlib.Path(__file__).parent.resolve().joinpath('ray_results')

ray.shutdown() # Kill Ray incase it didn't stop cleanly in another run
ray.init(local_mode=local_mode) # set true for better debugging, but need be false for scaling up

# If you need multiple callbacks, build them with this function:
# callback_list = [WinRatioCallback, 
#                 # This one needs a lambda to sneak our own argument in
#                 lambda: FlapActionMetricCallback(report_hist=False),
#                 lambda: SaveReplayCallback(replay_config={
#                     'replay_folder': replay_folder,
#                     'record_mode': 'off',
#                 })
#             ]

# callbacks = make_multi_callbacks(callback_list)

unit_types = ['marine', 'marauder', 'medivac']
map_name = '10gen_terran'
capability_config = {
    #FIXME: setup a config for generating agents and starting positions
}

def policy_map_fn(agent_id: str, *args, **kwargs):
    """Maps policy to name of unit type"""
    return None # FIXME: Implement this function

# NOTE: _enable_new_api_stack is set to false to force use actionmasking model instead of going to rl_modules which are in alpha stage
# For config options see https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
# Make sure to use the appropriate action masking model that pairs with the framework you are using
# You must setup the multi_agent() config to use the policy map func and set the policy names
# Add the win rate callback here as well
config = (  # Configure the algorithm
    AlgorithmConfig() # FIXME replace this with the config of a specific algorithm to use
    .environment(
        # FIXME put the environment class or registered name here, as well as an env_config dictionary
        )
    .experimental(_enable_new_api_stack=False) # Leave this set to false
    # FIXME populate the rest of the config with more methods below this
)

# Make callbacks for the tuner
run_callbacks = []
if use_wandb:
    run_callbacks.append(WandbLoggerCallback(project='masc2'))

tuner = tune.Tuner(
    # FIXME setup the tuner to run the experiment with the config you made
    # Be sure to setup the paths and config for saving checkpoints and stopping criteria
)

results = tuner.fit()
