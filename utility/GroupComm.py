import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .basics import *
from .DPRNN import *
from .TCN import *
from .DPTNet import *
from .sudo_rm_rf import *

# GroupComm-RNN
class GC_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', num_group=2, dropout=0, num_layers=1, bidirectional=False):
        super(GC_RNN, self).__init__()
        
        self.TAC = nn.ModuleList([])
        self.rnn = nn.ModuleList([])
        self.LN = nn.ModuleList([])
        
        self.num_layers = num_layers
        self.num_group = num_group
        
        for i in range(num_layers):
            self.TAC.append(TAC(input_size//num_group, hidden_size*3//num_group))
            self.rnn.append(ProjRNN(input_size//num_group, hidden_size//num_group, rnn_type, dropout, bidirectional))
            self.LN.append(nn.GroupNorm(1, input_size//num_group))

    def forward(self, input):
        # input shape: batch, dim, seq_len
        # split into groups
        batch_size, dim, seq_len = input.shape
        
        output = input.view(batch_size, self.num_group, -1, seq_len)
        for i in range(self.num_layers):
            # GroupComm via TAC
            output = self.TAC[i](output).transpose(2,3).contiguous()  # B, G, L, N
            output = output.view(batch_size*self.num_group, seq_len, -1)  # B*G, L, dim
            
            # RNN
            rnn_output = self.rnn[i](output)
            norm_output = self.LN[i](rnn_output.transpose(1,2))
            output = output + norm_output.transpose(1,2)  # B*G, L, dim
            output = output.view(batch_size, self.num_group, seq_len, -1).transpose(2,3).contiguous()  # B, G, dim, L
            
        output = output.view(batch_size, dim, seq_len)
        
        return output
    
# GroupComm-DPRNN
class GC_DPRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_group=2,
                 dropout=0, num_layers=1, bidirectional=True):
        super(GC_DPRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_group = num_group
        self.num_spk = output_size // input_size
        self.factor = int(bidirectional)+1
        
        # dual-path RNN with TAC
        self.TAC = nn.ModuleList([])
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.TAC.append(TAC(input_size//num_group, hidden_size*3//num_group))
            self.row_rnn.append(ProjRNN(input_size//num_group, hidden_size//num_group, 'LSTM', dropout, bidirectional=True))
            self.col_rnn.append(ProjRNN(input_size//num_group, hidden_size//num_group, 'LSTM', dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size//num_group, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size//num_group, eps=1e-8))
            
        self.output = nn.Conv2d(input_size//num_group, output_size//num_group, 1)
            
    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        
        batch_size, N, dim1, dim2 = input.shape
        output = input.view(batch_size, self.num_group, -1, dim1, dim2) 
        
        for i in range(len(self.row_rnn)):
            
            # GroupComm
            output = self.TAC[i](output.view(batch_size, self.num_group, -1, dim1*dim2))  # B, G, N/G, dim1*dim2
            output = output.view(batch_size*self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2
            
            # intra-block
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*self.num_group*dim2, dim1, -1)  # B*G*dim2, dim1, N/G
            row_output = self.row_rnn[i](row_input)  # B*G*dim2, dim1, N/G
            row_output = row_output.view(batch_size*self.num_group, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B*G, N/G, dim1, dim2
            row_output = self.row_norm[i](row_output.view(batch_size*self.num_group, -1, dim1, dim2)).view(output.shape)  # B*G, N/G, dim1, dim2
            output = output + row_output  # B*G, N/G, dim1, dim2
            
            # inter-block
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*self.num_group*dim1, dim2, -1)  # B*G*dim1, dim2, N/G
            col_output = self.col_rnn[i](col_input)  # B*G*dim1, dim2, N/G
            col_output = col_output.view(batch_size*self.num_group, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B*G, N/G, dim1, dim2
            col_output = self.col_norm[i](col_output.view(batch_size*self.num_group, -1, dim1, dim2)).view(output.shape)  # B*G, N/G, dim1, dim2
            output = output + col_output  # B*G, N/G, dim1, dim2
        
        output = output.view(batch_size*self.num_group, -1, dim1, dim2)
        output = self.output(output).view(batch_size, self.num_group, self.num_spk, -1, dim1, dim2)
        output = output.transpose(1,2).contiguous()
            
        return output

# GroupComm-TCN
class GC_TCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, 
                 causal=False, dilated=True, num_group=2):
        super(GC_TCN, self).__init__()
        
        # input is a sequence of features of shape (B, N, L)
        
        # TCN
        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group
        
        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(input_dim//num_group, hidden_dim//num_group, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal)) 
                else:
                    self.TCN.append(DepthConv1d(input_dim//num_group, hidden_dim//num_group, kernel, dilation=1, padding=1, skip=skip, causal=causal))  
                self.TAC.append(TAC(input_dim//num_group, hidden_dim*3//num_group))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
                    
        #print("Receptive field: {:3d} frames.".format(self.receptive_field))
        
        # output layer
        
        self.output = nn.Conv1d(input_dim//num_group, output_dim//num_group, 1)
        
        self.skip = skip
        
    def forward(self, input):
        
        # input shape: (B, N, L)
        batch_size, N, L = input.shape
        output = input.view(batch_size, self.num_group, -1, L)  # B, G, N/G, L
        
        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)  # B, G, N/G, L
                output = output.view(batch_size*self.num_group, -1, L)  # B*G, N/G, L
                residual, skip = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)  # B, G, N/G, L
                skip_connection = skip_connection + skip  # B*G, N/G, L
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)  # B, G, N/G, L
                output = output.view(batch_size*self.num_group, -1, L)  # B*G, N/G, L
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)  # B, G, N/G, L
            
        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output.view(batch_size*self.num_group, -1, L))
            
        output = output.view(batch_size, -1, L)  # B, N, L
        
        return output

    
# GroupComm-UBlock
class GC_UConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    '''

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, num_group=16):
        super(GC_UConvBlock, self).__init__()
        
        self.num_group = num_group
        self.TAC = TAC(out_channels//num_group, out_channels*3//num_group)
        self.UBlock = UConvBlock(out_channels//num_group, in_channels//num_group, upsampling_depth=upsampling_depth)

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        batch_size, N, L = x.shape
        
        # TAC across groups
        output = self.TAC(x.view(batch_size, self.num_group, -1, L)).view(batch_size*self.num_group, -1, L)
        # UBlock for each grouo
        output = self.UBlock(output)

        return output.view(batch_size, N, L)
    
# GroupComm-DPTNet
class GC_DPTNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_group=16,
                 dropout=0, num_layers=1):
        super(GC_DPTNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_group = num_group
        self.num_spk = output_size // input_size
        
        # dual-path Transformer
        self.row_xfmr = nn.ModuleList([])
        self.col_xfmr = nn.ModuleList([])
        self.TAC = nn.ModuleList([])
        for i in range(num_layers):
            self.TAC.append(TAC(input_size//num_group, hidden_size*3//num_group))
            self.row_xfmr.append(SingleTransformer(input_size=input_size//num_group, dim_feedforward=hidden_size//num_group))
            self.col_xfmr.append(SingleTransformer(input_size=input_size//num_group, dim_feedforward=hidden_size//num_group))
            
        self.output = nn.Conv2d(input_size//num_group, output_size//num_group, 1)
            
    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        
        batch_size, N, dim1, dim2 = input.shape
        output = input.view(batch_size, self.num_group, -1, dim1, dim2)
        
        for i in range(len(self.row_xfmr)):
            
            # GroupComm
            output = self.TAC[i](output.view(batch_size, self.num_group, -1, dim1*dim2))  # B, G, N/G, dim1*dim2
            output = output.view(batch_size*self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2
            
            row_input = output.permute(0,3,2,1).contiguous().view(batch_size*self.num_group*dim2, dim1, -1)  # B*G*dim2, dim1, N/G
            row_output = self.row_xfmr[i](row_input)  # B*G*dim2, dim1, H
            row_output = row_output.view(batch_size*self.num_group, dim2, dim1, -1).permute(0,3,2,1).contiguous()  # B*G, N, dim1, dim2
            output = output + row_output
            
            col_input = output.permute(0,2,3,1).contiguous().view(batch_size*self.num_group*dim1, dim2, -1)  # B*G*dim1, dim2, N
            col_output = self.col_xfmr[i](col_input)  # B*G*dim1, dim2, H
            col_output = col_output.view(batch_size*self.num_group, dim1, dim2, -1).permute(0,3,1,2).contiguous()  # B*G, N, dim1, dim2
            output = output + col_output
            
        output = output.view(batch_size*self.num_group, -1, dim1, dim2)
        output = self.output(output).view(batch_size, self.num_group, self.num_spk, -1, dim1, dim2)
        output = output.transpose(1,2).contiguous()
            
        return output
    
# wrapper for dual-path models
class DP_wrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_spk=2, num_group=16,
                 layer=4, block_size=100, bidirectional=True, module='RNN'):
        super(DP_wrapper, self).__init__()
        
        assert module in ['RNN', 'Transformer'], "Only RNN and Transformer are supported now."
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer = layer
        self.block_size = block_size
        self.num_spk = num_spk
        self.num_group = num_group
        
        self.eps = 1e-8
        
        if module == 'RNN':
            self.seq_model = GC_DPRNN(self.input_dim, self.hidden_dim, self.output_dim*self.num_spk, 
                                      num_group=self.num_group, num_layers=layer, bidirectional=bidirectional)
        elif module == 'Transformer':
            self.seq_model = GC_DPTNet(self.input_dim, self.hidden_dim, self.output_dim*self.num_spk, 
                                       num_group=self.num_group, num_layers=layer)
    
    def forward(self, input):
        
        batch_size = input.shape[0]
        
        # split the input into overlapped, longer segments
        input_blocks, input_rest = split_feature(input, self.block_size)  # B, N, L, K
        
        # pass to sequence modeling model
        output = self.seq_model(input_blocks).view(batch_size*self.num_spk, 
                                                   self.input_dim, self.block_size, -1)  # B, C, N, L, K
        
        # overlap-and-add of the outputs
        output = merge_feature(output, input_rest)
        output = output.view(batch_size, self.num_spk, self.output_dim, -1)  # B, C, K, T
        
        return output