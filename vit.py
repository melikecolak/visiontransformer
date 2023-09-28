# pip install einops
# pip install vit-pytorch

import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.optim as optim
from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from collections import OrderedDict
import os, csv, time
import matplotlib.pyplot as plt
import numpy as np



'''------------------------- Patching -----------------------------------'''
class Patching(nn.Module):
    def __init__(self, in_channels= 3, img_size = 224, patch_size= 16, embed_size = 768):
      # embed_size = in_channels x patchsize**2
        super(Patching, self).__init__()

        self.patch_size = patch_size
        self.num_path = int(img_size//patch_size)**2
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size),
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, embed_size ),
            nn.LayerNorm(embed_size ))

        self.class_token = nn.Parameter(torch.randn(1,1, embed_size))  
        self.pos_embedding = nn.Parameter(torch.randn(self.num_path + 1, embed_size))

    def forward(self, x):
        b, c, h, w = x.shape
        # x = rearrange(x,'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.projection(x)
        class_token = repeat(self.class_token, '() n e -> b n e', b=b)

        x = torch.cat([class_token, x], dim=1)
        # add position embedding
        x += self.pos_embedding

        return x

# x = torch.rand(1,3,224,224)   
# Patching()(x).shape
# model = Patching()
# summary(model, (3,224,224))

'''------------------------- Attention and MultilHeadAttention -----------------------------------'''

class MultiHeadAttention(nn.Module):

  def __init__(self, embed_size, num_heads, dropout = 0):

    super(MultiHeadAttention, self).__init__()

    self.emb_size = embed_size
    self.num_heads = num_heads
    self.head_size = embed_size//num_heads

    assert embed_size % num_heads == 0, "embed_size % num_heads should be zero."

    # Determin Wq, Qk and Qv in Attention
    self.keys = nn.Linear(embed_size, self.head_size*num_heads) # (Wk matrix) 
    self.queries = nn.Linear(embed_size,  self.head_size*num_heads) # (Wq matrix) 
    self.values = nn.Linear(embed_size,  self.head_size*num_heads) # (Wv matrix) 

    self.att_drop = nn.Dropout(dropout)
    self.dense = nn.Linear(embed_size, embed_size)

  def forward(self, x):     
    # x.shape = [Batchsize (B) x num_patch (np) x embed_size (ez)] 
    batch_size, np, ez = x.shape
    key = self.keys(x)            # [Bx (np x ez)] x [ez x ez] = [B x np x ez] 
    query = self.queries(x)       # [Bx (np x ez)] x [ez x ez] = [B x np x ez]
    value = self.values(x)        # [Bx (np x ez)] x [ez x ez] = [B x np x ez]

    # split key, query and value in many num_heads
    key = key.view(batch_size, -1, self.num_heads, self.head_size)      # [B x np x h x s]
    query = query.view(batch_size, -1, self.num_heads, self.head_size)  # [B x np x h x s]
    value = value.view(batch_size, -1, self.num_heads, self.head_size)  # [B x np x h x s]

    key = key.permute(2, 0, 1 ,3).contiguous().view(batch_size * self.num_heads, -1, self.head_size) # [(Bh) x np x s]
    query = query.permute(2, 0, 1 ,3).contiguous().view(batch_size * self.num_heads, -1, self.head_size) # [(Bh) x np x s]
    value = value.permute(2, 0, 1 ,3).contiguous().view(batch_size * self.num_heads, -1, self.head_size) # [(Bh) x np x s]
    # Q x K matrix
    score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.head_size)
    soft = F.softmax(score, -1)
    context = torch.bmm(soft, value)
    context = self.att_drop(context)
    # Convert to the original size
    context = context.view(self.num_heads, batch_size, -1, self.head_size) # [h x B x np x s]
    context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_size)

    attention = self.dense(context)

    return attention #  [Batchsize (B) x num_patch (np) x embed_size (ez)]

# x = torch.rand(1,4 ,32)
# attention = MultiHeadAttention( embed_size=32, num_heads=2)
# summary(attention, (4, 32))

'''
with embed_size=32, num_heads=2
[1,4,32] x W (linear)---> [1,4,32] ---(devide by 2 heads)----> [1 2 4 16] shape of Q, K, V
Soft = QxK [1 2 4 16] x [1 2 16 4] = [1 2 4 4] 
attention = [1 2 4 4] --- x([1 2 4 16])---> [1 2 4 16] ---rearrange---> [1, 4, 32] ---(dense)---> [1, 4, 32]
'''

'''------------------------- Transformer Block -----------------------------------'''
class TransformerBlock(nn.Module):
  def __init__(self, embed_size, num_heads, expansion, dropout = 0):
    super(TransformerBlock, self).__init__()

    self.norm1 = nn.LayerNorm(embed_size)
    self.mul_attention = MultiHeadAttention(embed_size,num_heads)
    self.drop = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(embed_size)
    self.mlp = nn.Sequential(nn.Linear(embed_size, expansion*embed_size),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(expansion*embed_size, embed_size))
                            
  def forward(self, x):
    out = x + self.drop(self.mul_attention(self.norm1(x)))
    out = out + self.drop(self.mlp(self.norm2(out)))
    return out

# x = torch.rand(1,4 ,32)
# block = TransformerBlock(embed_size =32, num_heads=2, expansion=2)
# print(block(x).shape)
# summary(block, (4, 32))

'''------------------------- Encoder -----------------------------------'''

class Encoder(nn.Module):
  def __init__(self,embed_size, num_heads, expansion, dropout, depth):
     super(Encoder, self).__init__()

     layers: OrderedDict[str, nn.Module] = OrderedDict()

     for i in range(depth):
       layers[f"encoder_layer_{i}"] = TransformerBlock(embed_size, num_heads, expansion, dropout)
     self.layers = nn.Sequential(layers)
  
  def forward(self, x):
    return self.layers(x)

# x = torch.rand(1,4 ,32)   
# encoder = Encoder(embed_size=32, num_heads=2, expansion=2, dropout=0.2, depth=2)
# print(encoder)
# print(encoder(x).shape)
# summary(encoder, (4, 32)) 


'''------------------------- Vision Transfomer Model-----------------------------------'''
class VIT(nn.Module):
  def __init__(self,  in_channels= 3, img_size = 32, patch_size= 4, embed_size = 48, 
               num_heads = 2, expansion = 4, dropout= 0.2, depth = 4, num_classes = 10):
    # embed_size = in_channels x patchsize**2
    super(VIT, self).__init__()
    self.path_embedding = Patching(in_channels, img_size, patch_size, embed_size) 
    self.encoder = Encoder(embed_size, num_heads, expansion, dropout, depth)
    self.num_class = nn.Sequential(Reduce('b n e -> b e', reduction='mean'), 
                                   nn.LayerNorm(embed_size), 
                                   nn.Linear(embed_size, num_classes))

  def forward(self, x):
    x = self.path_embedding(x)
    x = self.encoder(x)
    x = self.num_class(x)

    return x


model = VIT()
# x = torch.rand(1,3,32, 32)
# summary(model, (3,32, 32))

