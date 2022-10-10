import unittest
from nn_model.test_transformers import TestTransformerModel
from nn_model.test_whole_sequence_model import TestWholeSequenceModel
from util.test_util import TestUtils
from policy.test_policy import TestPolicy
import gym

if __name__ == '__main__':
    gym.logger.set_level(50) #damn
    unittest.main()