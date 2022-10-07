import unittest
import torch as th
from active_critic.utils.dataset import DatasetAC

class TestDataset(unittest.TestCase):

    def test_dataset_cuda(self):
        device = 'cuda'
        dataset = DatasetAC(device=device)

        batch_size = 5
        seq_len = 5
        action_dim = 3
        obs_dim = 4

        obsv = th.ones(size=[batch_size, seq_len, obs_dim], device=device)
        actions = th.ones(size=[batch_size, seq_len, action_dim], device=device)
        actions[:2, -2:] = 0
        reward = th.ones(size=[batch_size, seq_len], dtype=th.bool, device=device)
        reward[:2, -2:] = 0

        dataset.set_data(obsv=obsv, actions=actions, reward=reward)

        self.assertTrue(th.equal(dataset.success, reward[:,-1] == 1), 'Success is not correctly calulated')
        
        loader = th.utils.data.DataLoader(dataset=dataset, batch_size=32)

        dataset.onyl_positiv = True

        for data in loader:
            obsv, actions, reward = data
            self.assertTrue(list(obsv.shape) == [batch_size-2, seq_len, obs_dim])
            self.assertTrue(list(actions.shape) == [batch_size-2, seq_len, action_dim])
            self.assertTrue(list(reward.shape) == [batch_size-2, seq_len])
            self.assertTrue(th.all(actions == 1))

        dataset.onyl_positiv = False

        for data in loader:
            obsv, actions, reward = data
            assert list(obsv.shape) == [batch_size, seq_len, obs_dim]
            assert list(actions.shape) == [batch_size, seq_len, action_dim]
            assert list(reward.shape) == [batch_size, seq_len]
            
        dataset.add_data(obsv=obsv, actions=actions, reward=reward)

        dataset.onyl_positiv = True

        for data in loader:
            obsv, actions, reward = data
            assert list(obsv.shape) == [2*(batch_size-2), seq_len, obs_dim]
            assert list(actions.shape) == [2*(batch_size-2), seq_len, action_dim]
            assert list(reward.shape) == [2*(batch_size-2), seq_len]
            assert th.all(actions == 1)

        dataset.onyl_positiv = False

        for data in loader:
            obsv, actions, reward = data
            assert list(obsv.shape) == [2*batch_size, seq_len, obs_dim]
            assert list(actions.shape) == [2*batch_size, seq_len, action_dim]
            assert list(reward.shape) == [2*batch_size, seq_len]

    def test_dataset_cpu(self):
        device = 'cpu'
        dataset = DatasetAC(device=device)

        batch_size = 5
        seq_len = 5
        action_dim = 3
        obs_dim = 4

        inpt = th.ones(size=[batch_size, seq_len, obs_dim], device=device)
        label = th.ones(size=[batch_size, seq_len, action_dim], device=device)
        label[:2, -2:] = 0
        reward = th.ones(size=[batch_size, seq_len], dtype=th.bool, device=device)
        reward[:2, -2:] = 0

        dataset.set_data(obsv=inpt, actions=label, reward=reward)

        self.assertTrue(th.equal(dataset.success, reward[:,-1] == 1), 'Success is not correctly calulated')
        
        loader = th.utils.data.DataLoader(dataset=dataset, batch_size=32)

        dataset.onyl_positiv = True

        for data in loader:
            inpt, label, reward = data
            self.assertTrue(list(inpt.shape) == [batch_size-2, seq_len, obs_dim])
            self.assertTrue(list(label.shape) == [batch_size-2, seq_len, action_dim])
            self.assertTrue(list(reward.shape) == [batch_size-2, seq_len])
            self.assertTrue(th.all(label == 1))

        dataset.onyl_positiv = False

        for data in loader:
            inpt, label, reward = data
            assert list(inpt.shape) == [batch_size, seq_len, obs_dim]
            assert list(label.shape) == [batch_size, seq_len, action_dim]
            assert list(reward.shape) == [batch_size, seq_len]
            
        dataset.add_data(obsv=inpt, actions=label, reward=reward)

        dataset.onyl_positiv = True

        for data in loader:
            inpt, label, reward = data
            assert list(inpt.shape) == [2*(batch_size-2), seq_len, obs_dim]
            assert list(label.shape) == [2*(batch_size-2), seq_len, action_dim]
            assert list(reward.shape) == [2*(batch_size-2), seq_len]
            assert th.all(label == 1)

        dataset.onyl_positiv = False

        for data in loader:
            inpt, label, reward = data
            assert list(inpt.shape) == [2*batch_size, seq_len, obs_dim]
            assert list(label.shape) == [2*batch_size, seq_len, action_dim]
            assert list(reward.shape) == [2*batch_size, seq_len]

if __name__ == '__main__':
    unittest.main()