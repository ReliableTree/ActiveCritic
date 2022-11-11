import unittest
import torch as th
from active_critic.model_src.state_model import *

class TestStateModel(unittest.TestCase):

    def make_smargs(self):
        output_dim = 2
        smargs = StateModelArgs()
        smargs.arch = [100, 100, output_dim]
        smargs.device = 'cpu'
        smargs.lr = 1e-2
        return smargs, output_dim

    def make_sm_data(self, output_dim:int):
        batch_size = 2
        seq_len = 5
        inpt_dim = 3
        inpt = th.rand([batch_size, seq_len, inpt_dim])
        label = th.rand([batch_size, seq_len, output_dim])
        return inpt, label, batch_size, seq_len, inpt_dim

    def test_state_model(self):
        smargs, output_dim = self.make_smargs()
        inpt, label, batch_size, seq_len, inpt_dim = self.make_sm_data(output_dim=output_dim)

        sm = StateModel(args=smargs)
        th.manual_seed(0)
        output = sm.forward(inpt)
        output.shape
        expected_size = [batch_size, seq_len, output_dim]
        self.assertTrue(list(output.shape) == expected_size, 'Output size not as expected.')
        for i in range(1000):
            loss = sm.calc_loss(inpt=inpt, label=label)
            sm.optimizer.zero_grad()
            loss.backward()
            sm.optimizer.step()
        self.assertTrue(loss < 1e-5, 'Loss did not converge.')
        sm.init_model()
        loss = sm.calc_loss(inpt=inpt, label=label)
        self.assertTrue(loss > 1e-2, 'Reset did not change the network.')
        for i in range(1000):
            loss = sm.calc_loss(inpt=inpt, label=label)
            sm.optimizer.zero_grad()
            loss.backward()
            sm.optimizer.step()
        self.assertTrue(loss < 1e-5, 'No convergence after reset.')

if __name__ == '__main__':
    unittest.main()