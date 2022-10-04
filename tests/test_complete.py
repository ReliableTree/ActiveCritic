from unicodedata import name
import unittest
from tests.nn_model.test_transformers import TestTransformerModel
from tests.nn_model.test_whole_sequence_model import TestWholeSequenceModel
from tests.util.test_util import TestUtils
from tests.policy.test_policy import TestPolicy
import gym

if __name__ == '__main__':
    gym.logger.set_level(50) #damn
    unittest.main()