import numpy as np
import torch as th

class DummyExtractor:
    def __init__(self):
        pass

    def forward(self, features):
        if type(features) is np.ndarray:
            features = th.tensor(features)
        return features

def new_epoch(current_obs, check_obsvs):
    result =  not th.equal(current_obs.reshape(-1)[-3:], check_obsvs.reshape(-1)[-3:])
    return result