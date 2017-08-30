import torch
import torch.nn as nn

class AME(nn.Module):
    def __init__(self, ninp1, nhid1, ninp2, nhid2):
        super(AME, self).__init__()

        self.lstm1 = nn.LSTMCell(ninp1, nhid1)
        self.lstm21 = nn.LSTMCell(ninp2, nhid2)
        self.lstm22 = nn.LSTMCell(ninp2, nhid2)

    def forward(self, input, h_00, h_10, enc=False):
        h_01, c_01 = self.lstm1(input, h_00)
        h_11, c_11 = self.lstm21(h_01, h_10)
        if enc: 
            h_enc, c_enc = self.lstm22(h_01, enc)
            return h_11, h_enc, (h_01, c_01), (h_11, c_11), (h_enc, c_enc)
            
        return h_11, (h_01, c_01), (h_11, c_11)
