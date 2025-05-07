import math

import numpy as np
# import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.init
from torch.nn.parallel.distributed import DistributedDataParallel
import copy
import torch
import torch.nn as nn
import torch.nn.init
import torchtext

# =========================
# Text feature extraction
# =========================
class BiGRUModel(nn.Module):
    def __init__(self, word2idx, embed_dim=512, word_dim=300, num_layers=1, use_bidirectional_rnn=True):
        super(BiGRUModel, self).__init__()
        self.gpuid = 0,1
        self.embed_dim = embed_dim
        self.word_dim = word_dim
        self.vocab_size = 8590
        self.num_layers = num_layers
        self.use_bidirectional_rnn = use_bidirectional_rnn
        # word embedding
        self.embed = nn.Embedding(self.vocab_size, self.word_dim)

        # caption embedding
        self.use_bidirectional_rnn = self.use_bidirectional_rnn
        print('=> using bidirectional rnn:{}'.format(self.use_bidirectional_rnn))
        self.rnn = nn.GRU(self.word_dim, self.embed_dim, self.num_layers,
                          batch_first=True, bidirectional=self.use_bidirectional_rnn)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(0.4)

        self.init_weights(word2idx, self.word_dim)

    def init_weights(self, word2idx, word_dim):
        # Load pretrained word embedding
        wemb = torchtext.vocab.GloVe()

        assert wemb.vectors.shape[1] == word_dim

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace(
                    '-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # Embed word ids to vectors
        x = self.dropout(self.embed(x))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_rnn:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] +
                       cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        return cap_emb