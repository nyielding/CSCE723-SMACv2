from typing import Dict
from gymnasium import Env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID

class WinRatioCallback(DefaultCallbacks):
    """
    Saves at the end of an episode if the battle was won or not as a custom metric. 
    This will make the metrics for min/mean/max battle win ratio available in the tensorboard or wandb. 
    Note that only the mean is relevant, since min will be 0.0 and max 1.0.
    """
    def on_episode_end(self, *, episode, env_runner = None, metrics_logger = None, env = None, env_index: int, rl_module: RLModule | None = None, worker: EnvRunner | None = None, base_env: BaseEnv | None = None, policies: Dict[str, Policy] | None = None, **kwargs) -> None:
        pass # FIXME: Implement this method
