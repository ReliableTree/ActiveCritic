from active_critic.model_src.whole_sequence_model import WholeSequenceModel
from active_critic.model_src.transformer import ModelSetup, TransformerModel
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


if __name__ == '__main__':
    unittest.main()