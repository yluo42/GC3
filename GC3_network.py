from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

from utility.basics import *
from utility.GroupComm import *

class TasNet(nn.Module):
    def __init__(self, enc_dim=128, hidden_dim=256, sr=16000, win=2,
                 layer=6, num_spk=2, context_size=32, 
                 group_size=16, block_size=24, module='DPRNN'):
        super(TasNet, self).__init__()
        
        assert module in ['DPRNN', 'TCN', 'UBlock', 'Transformer'], "Only DPRNN, TCN, UBlock and Transformer are supported now."
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        self.group_size = group_size
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.norm = nn.GroupNorm(1, self.enc_dim)
        
        # context encoder/decoder
        self.context_enc = GC_RNN(self.enc_dim, self.hidden_dim, num_group=self.group_size, num_layers=2, bidirectional=True)
        self.context_dec = GC_RNN(self.enc_dim, self.hidden_dim, num_group=self.group_size, num_layers=2, bidirectional=True)
        
        # sequence modeling
        if module == 'DPRNN':
            self.seq_model = DP_wrapper(self.enc_dim, self.hidden_dim, self.enc_dim, 
                                        num_spk=1, num_group=self.group_size, layer=layer, 
                                        block_size=block_size, module='RNN')
        elif module == 'TCN':
            self.seq_model = GC_TCN(self.enc_dim, self.enc_dim, self.enc_dim*4,
                                    layer=layer, stack=2, kernel=3, causal=False, 
                                    num_group=self.group_size)
        elif module == 'UBlock':
            self.seq_model = nn.Sequential(*[GC_UConvBlock(out_channels=self.enc_dim,
                                                           in_channels=self.hidden_dim*2,
                                                           upsampling_depth=5,
                                                           num_group=self.group_size)
                                             for _ in range(layer)])
        elif module == 'Transformer':
            self.seq_model = DP_wrapper(self.enc_dim, self.hidden_dim, self.enc_dim, 
                                        num_spk=1, num_group=self.group_size, layer=layer, 
                                        block_size=block_size, module='Transformer')

        # mask estimation layer
        self.mask = nn.Sequential(nn.Conv1d(self.enc_dim//self.group_size, self.enc_dim*self.num_spk//self.group_size, 1),
                                  nn.ReLU(inplace=True)
                                 )
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)
        
    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest
    
    def forward(self, input):
        
        # padding
        output, rest = self.pad_input(input, self.win)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output.unsqueeze(1))  # B, N, T
        seq_len = enc_output.shape[-1]
        enc_feature = self.norm(enc_output)
        
        # context encoding
        squeeze_block, squeeze_rest = split_feature(enc_feature, self.context_size)  # B, N, context, L
        squeeze_frame = squeeze_block.shape[-1]
        squeeze_input = squeeze_block.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame, self.enc_dim, 
                                                                         self.context_size)  # B*L, N, context
        squeeze_output = self.context_enc(squeeze_input)  # B*L, N, context
        squeeze_mean = squeeze_output.mean(2).view(batch_size, squeeze_frame, 
                                                   self.enc_dim).transpose(1,2).contiguous()  # B, N, L
        
        # sequence modeling
        feature_output = self.seq_model(squeeze_mean).view(batch_size, -1, squeeze_frame)  # B, N, L
        
        # context decoding
        feature_output = feature_output.unsqueeze(2) + squeeze_block  # B, N, context, L
        feature_output = feature_output.permute(0,3,1,2).contiguous().view(batch_size*squeeze_frame,
                                                                           self.enc_dim,
                                                                           self.context_size)  # B*L, N, context
        unsqueeze_output = self.context_dec(feature_output).view(batch_size, squeeze_frame, 
                                                                 self.enc_dim, -1)  # B, L, N, context
        unsqueeze_output = unsqueeze_output.permute(0,2,3,1).contiguous()  # B, N, context, L
        unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T
        
        # mask estimation
        unsqueeze_output = unsqueeze_output.view(batch_size*self.group_size, -1, unsqueeze_output.shape[-1])
        mask = self.mask(unsqueeze_output).view(batch_size, self.group_size, self.num_spk, 
                                                self.enc_dim//self.group_size, -1)  # B, G, C, N/G, T
        mask = mask.transpose(1,2).contiguous().view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, T
        output = mask * enc_output.unsqueeze(1)  # B, C, N, T
        
        # waveform decoder
        output = self.decoder(output.view(batch_size*self.num_spk, self.enc_dim, seq_len))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)
        
        return output
    
def test_model(model):
    x = torch.rand(2, 64000)  # (batch, length)
    y = model(x)
    print(y.shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model_DPRNN = TasNet(module='DPRNN')
    model_TCN = TasNet(module='TCN')
    model_UBlock = TasNet(module='UBlock')
    model_DPTNet = TasNet(module='Transformer')

    test_model(model_DPRNN)
    test_model(model_TCN)
    test_model(model_UBlock)
    test_model(model_DPTNet)
