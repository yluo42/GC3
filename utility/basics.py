import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ProjRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', dropout=0, bidirectional=False):
        super(ProjRNN, self).__init__()
        
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size*self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return proj_output
    
# transform-average-concatenate (TAC)
class TAC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()
        
        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.PReLU()
                                      )
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.PReLU()
                                     )
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size*2, input_size),
                                        nn.PReLU()
                                       )
        self.TAC_norm = nn.GroupNorm(1, input_size)
        
    def forward(self, input):
        # input shape: batch, group, N, seq_length
        
        batch_size, G, N, T = input.shape
        output = input
        
        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0,3,1,2).contiguous().view(-1, N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H
        
        # mean pooling
        group_mean = group_output.mean(2).view(batch_size*T, -1)  # B*T, H
        
        # concate
        group_output = group_output.view(batch_size*T, G, -1)  # B*T, G, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0,2,3,1).contiguous()  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size*G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)
        
        return output
    
def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, block_stride)).type(input.type()).to(input.device)
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest

def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = input[:,:,:-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    block2 = input[:,:,block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    block = torch.cat([block1, block2], 3).view(batch_size, dim, -1, block_size).transpose(2, 3)

    return block.contiguous(), rest

def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size*2)  # B, N, K, L

    input1 = input[:,:,:,:block_size].contiguous().view(batch_size, dim, -1)[:,:,block_stride:]
    input2 = input[:,:,:,block_size:].contiguous().view(batch_size, dim, -1)[:,:,:-block_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:,:,:-rest]

    return output.contiguous()  # B, N, T