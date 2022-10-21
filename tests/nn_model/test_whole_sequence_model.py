from active_critic.model_src.whole_sequence_model import WholeSequenceModel, OptimizeMaximumCritic
from active_critic.model_src.transformer import ModelSetup, TransformerModel
from matplotlib.pyplot import close
import torch as th
from active_critic.utils.test_utils import make_mask_data, make_seq_encoding_data, make_critic_data, make_wsm_setup
import unittest


class TestWholeSequenceModel(unittest.TestCase):

    def test_WholeSequenceActor(self):
        th.manual_seed(0)

        seq_len = 6
        ntoken = 3
        d_output = 2
        batch_size = 2
        d_intput = 3

        wsa_setup = make_wsm_setup(seq_len=seq_len, d_output=d_output)

        wsa = WholeSequenceModel(wsms=wsa_setup)
        input = th.ones([batch_size, seq_len, d_intput],
                        dtype=th.float, device='cuda')
        output = wsa.forward(inputs=input)
        shape = output.shape

        self.assertTrue(shape[0] == batch_size)
        self.assertTrue(shape[1] == seq_len)
        self.assertTrue(shape[2] == d_output)

        inpt_seq, outpt_seq = make_seq_encoding_data(
            batch_size=batch_size, seq_len=seq_len, ntoken=ntoken, d_out=d_output)
        success = th.ones_like(inpt_seq, dtype=th.bool)
        wsa = WholeSequenceModel(wsms=wsa_setup)
        data = inpt_seq, outpt_seq, success
        for i in range(3000):
            res = wsa.optimizer_step(inputs=inpt_seq, label=outpt_seq)
        self.assertTrue(res['Loss '] < 1e-2,
                        'Actor did not converge.')

        res = wsa.optimizer_step(inputs=inpt_seq, label=outpt_seq, prefix='test')
        self.assertTrue('Loss test' in res)

        wsa.init_model()
        res = wsa.optimizer_step(inputs=inpt_seq, label=outpt_seq)
        self.assertTrue(res['Loss '] > 1e-1,
                        'Init Model did not cange the parameters.')
        for i in range(3000):
            res = wsa.optimizer_step(inputs=inpt_seq, label=outpt_seq)
        self.assertTrue(res['Loss '] < 1e-2,
                        'Did not converge after reinit.')

    def test_loss_fct(self):
        th.manual_seed(0)

        seq_len = 6
        ntoken = 3
        d_output = 2
        batch_size = 2
        d_intput = 3


        input = th.ones([batch_size, seq_len, 1],
                        dtype=th.float, device='cuda')
        input[0,0] = 3
        input[1,1] = 2
        org_input = th.clone(input)
        input.requires_grad = True

        goal = th.ones([batch_size, seq_len, 1],
                dtype=th.float, device='cuda')

        opt = th.optim.SGD([input], lr=1e-1)

        wsa_setup = make_wsm_setup(seq_len=seq_len, d_output=d_output)
        omc = OptimizeMaximumCritic(wsms=wsa_setup)
        loss = omc.loss_fct(result=input, label=goal)

        self.assertTrue((loss - 2.5).sum() == 0)
        loss.backward()
        opt.step()
        self.assertTrue(th.equal(input[0,1:], org_input[0,1:]))
        self.assertTrue(th.equal(input[1,0], org_input[1,0]))
        self.assertTrue(th.equal(input[1,2:], org_input[1,2:]))
        self.assertTrue(input[0,0] < org_input[0,0])
        self.assertTrue(input[1,1] < org_input[1,1])

    def test_tf_mask(self):
        batch_size = 2
        seq_len = 2
        obs_dim = 3
        device = 'cpu'

        inpt_seq, outpt_seq, mask = make_mask_data(batch_size,seq_len,obs_dim,device=device)
        wsms = make_wsm_setup(seq_len=seq_len, d_output=obs_dim, device=device)
        wsm = WholeSequenceModel(wsms=wsms)
        for i in range(1000):
            loss_dict = wsm.optimizer_step(inpt_seq, outpt_seq, tf_mask=mask)
        self.assertTrue(th.allclose(loss_dict['Loss '], th.tensor(0.125)))

        
if __name__ == '__main__':
    unittest.main()