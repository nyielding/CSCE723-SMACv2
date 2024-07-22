"""
This is a Command Line Interface (CLI) to evaluate a checkpoint with passed args.
May also be run as a script when changing default_checkpoint

For more info, in terminal from the workspace dir, type:
    python eval.py -h
"""

import time
import pathlib
import pprint
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from masc2.rllib.env import RLlibStarCraft2Env

# Fix tensorflow bug when using tf2 framework with Algorithm.from_checkpoint()
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

# # Put the checkpoint to run episodes on here, as well as number of workers and episodes to use. Looks something like this.
checkpoint = '/home/developer/masc2-arena/ray_results/SMAC_1/PPO_RLlibStarCraft2Env_d9a60_00000_0_2024-07-21_04-01-04/checkpoint_000004'
replay_folder = pathlib.Path(__file__).parent.joinpath('replays')
evaluation_duration = 10
evaluation_num_workers = 1
# # If changing the way units are spawned from last used in training, update config here
capability_config = {}

register_env("masc2", RLlibStarCraft2Env)

# Get the old config from the checkpoint
old_alg = Algorithm.from_checkpoint(checkpoint=checkpoint)
old_config = old_alg.get_config()
config = old_config.copy(copy_frozen=False) # make an unfrozen copy

# Update config for evaluation only run
config_update = {
    'env': 'masc2',
    'env_config': {
        'save_replays': True, 
        'replay_folder': replay_folder,
        'replay_duration': evaluation_duration,
        },
    'evaluation_config': {
        'evaluation_interval': 1,
        'evaluation_duration_unit': 'episodes',
        'evaluation_duration': int(evaluation_duration*2), # for some reason env does half the episodes in starcraft.py
        'evaluation_num_workers': evaluation_num_workers,
    },
    'num_rollout_workers': 0,
    # 'explore': False, # NOTE: DO NOT turn explore off with policy algs like PPO
}
if capability_config:
    config_update['env_config']['capability_config'] = capability_config

# Update config with dictionary
config.update_from_dict(config_update)

# Build new alg
alg: Algorithm = config.build()

# Restore the policy and training history
alg.restore(checkpoint_path=checkpoint)

# Run the evaluation
tic = time.perf_counter()
eval_results = alg.evaluate()

# Report how it went
win_rate = eval_results['env_runners']['custom_metrics'].get('battle_won_mean', None)
print('Win rate: ', win_rate)
print('Reward mean: ', eval_results['env_runners']['episode_reward_mean'])
print('Reward per episode: ')
pprint.pprint([x for x in enumerate(eval_results['env_runners']['hist_stats']['episode_reward'])])
pprint.pprint(eval_results)
seconds = time.perf_counter() - tic
mm, ss = divmod(seconds, 60)
hh, mm = divmod(mm, 60)
print(f'Total time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')
print('Evaluation duration: ', evaluation_duration)
print('Replay folder: ',replay_folder)
