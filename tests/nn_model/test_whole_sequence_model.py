from ActiveCritic.model_src.whole_sequence_model import WholeSequenceActor, WholeSequenceCritic, WholeSequenceModelSetup
from ActiveCritic.model_src.transformer import ModelSetup, TransformerModel, CriticTransformer
import torch as th
from ActiveCritic.tests.test_utils.utils import make_mask_data, make_seq_encoding_data, make_critic_data, make_wsm_setup
import unittest


class TestWholeSequenceModel(unittest.TestCase):

    def test_WholeSequenceActor(self):
        seq_len = 6
        ntoken = 3
        d_output = 2
        batch_size = 2
        d_intput = 3

        wsa_setup = make_wsm_setup(seq_len=seq_len, d_output=d_output, model_class=TransformerModel)

        wsa = WholeSequenceActor(wsms=wsa_setup)
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
        wsa = WholeSequenceActor(wsms=wsa_setup)
        data = inpt_seq, outpt_seq, success
        for i in range(3000):
            res = wsa.optimizer_step(data=data)
        self.assertTrue(res['Trajectory Loss '] < 1e-2,
                        'Actor did not converge.')

        res = wsa.optimizer_step(data=data, prefix='test')
        self.assertTrue('Trajectory Loss test' in res)

        wsa.init_model()
        res = wsa.optimizer_step(data=data)
        self.assertTrue(res['Trajectory Loss '] > 1e-1,
                        'Init Model did not cange the parameters.')
        for i in range(3000):
            res = wsa.optimizer_step(data=data)
        self.assertTrue(res['Trajectory Loss '] < 1e-2,
                        'Did not converge after reinit.')

    def test_WholeSequenceCritic(self):
        seq_len = 6
        ntoken = 3
        d_result = 1
        d_output = 2
        batch_size = 2
        inpt_seq, outpt_seq = make_critic_data(
            batch_size=batch_size, seq_len=seq_len, ntoken=ntoken)
        data = inpt_seq, None, outpt_seq

        wsa_setup = make_wsm_setup(seq_len=seq_len, d_output=d_output, model_class=CriticTransformer, d_result=d_result)

        model = WholeSequenceCritic(wsms=wsa_setup)
        for i in range(3000):
            res = model.optimizer_step(data)
        self.assertTrue(res['critic loss'] < 1e-2,
                        'total critic loss did not converge')
        self.assertTrue(res['critic loss positive'] < 1e-2,
                        'positive critic loss did not converge')
        self.assertTrue(res['critic loss negative'] < 1e-2,
                        'negative critic loss did not converge')

