import unittest
from active_critic.model_src.transformer import ModelSetup, TransformerModel, generate_square_subsequent_mask
import torch as th
from active_critic.utils.pytorch_utils import calcMSE
from active_critic.utils.test_utils import make_seq_encoding_data, make_mask_data, make_critic_data


class TestTransformerModel(unittest.TestCase):
    def test_seq_encoding(self):

        ms = ModelSetup()
        seq_len = 6
        batch_size = 4
        ntoken = 1
        d_output = 1
        ms.d_output = d_output
        ms.nhead = 1
        ms.d_hid = 10
        ms.d_model = 10
        ms.nlayers = 2
        ms.seq_len = seq_len
        ms.dropout = 0
        ms.ntoken = 1
        ms.device = 'cuda'

        inpt_seq, outpt_seq = make_seq_encoding_data(batch_size=batch_size, seq_len=seq_len, ntoken=ntoken, d_out=d_output)

        model = TransformerModel(model_setup=ms).to('cuda')
        with th.no_grad():
            model.forward(inpt_seq)
        optimizer = th.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = 1
        
        for i in range(3000):
            result = model.forward(inpt_seq)
            loss = calcMSE(result, outpt_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertTrue(loss < 1e-2, 'Could not converge to sequence. Maybe the positional encoding is wrong?')

    def test_mask(self):
        ms = ModelSetup()
        seq_len = 6
        batch_size = 4
        ntoken = 1
        d_output = 1
        ms.d_output = d_output
        ms.nhead = 1
        ms.d_hid = 10
        ms.d_model = 10
        ms.nlayers = 2
        ms.seq_len = seq_len
        ms.dropout = 0
        ms.ntoken = 1
        ms.lr = None
        ms.device = 'cuda'


        inpt_seq, outpt_seq, mask = make_mask_data(batch_size=batch_size, seq_len=seq_len, ntoken=ntoken)
        model = TransformerModel(model_setup=ms).to('cuda')
        with th.no_grad():
            model.forward(inpt_seq)
        optimizer = th.optim.Adam(params=model.parameters(), lr=1e-3)
        loss = 0
        for i in range(2000):
            result = model.forward(inpt_seq, mask=mask)
            loss = calcMSE(result, outpt_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertTrue(loss > 1e-1, 'Converged to masked knowledge.')

if __name__ == '__main__':
    unittest.main()