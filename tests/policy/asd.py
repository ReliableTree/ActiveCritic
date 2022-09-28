import gym
import numpy as np
import torch as th
from ActiveCritic.metaworld.metaworld.envs import \
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from ActiveCritic.model_src.transformer import (CriticTransformer, ModelSetup,
                                                TransformerModel)
from ActiveCritic.model_src.whole_sequence_model import (
    WholeSequenceActor, WholeSequenceCritic, WholeSequenceModelSetup)
from ActiveCritic.policy.active_critic_policy import (ACPOptResult,
                                                      ActiveCriticPolicy,
                                                      ActiveCriticPolicySetup)
from ActiveCritic.tests.test_utils.utils import make_wsm_setup
from ActiveCritic.utils.gym_utils import (DummyExtractor, make_policy_dict,
                                          new_epoch_reach)
from ActiveCritic.utils.pytorch_utils import make_partially_observed_seq
from gym.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
