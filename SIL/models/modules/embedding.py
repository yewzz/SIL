import torch
import torch.nn as nn
from models.modules.position import PositionalEncoding


class Embedding(nn.Module):
    """Words embeddings for encoder/decoder.
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        position_encoding (bool): see :obj:`modules.PositionalEncoding`
    """
    def __init__(self, word_vocab_size, word_vec_size,
        position_encoding=False, fix_word_embed=False, max_len=100):
        super(Embedding, self).__init__()

        self.word_vec_size = word_vec_size
        self.we = nn.Embedding(word_vocab_size, word_vec_size)
        if fix_word_embed:
            self.we.weight.requires_grad = False
        self.init_weight()

        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pe = PositionalEncoding(word_vec_size, dropout=0, max_len=max_len)

    def init_weight(self):
        std = 1. / (self.word_vec_size**0.5)
        nn.init.uniform_(self.we.weight, -std, std)

    def forward(self, word_idxs, step=None):
        """Computes the embeddings for words.
        Args:
            word_idxs (`LongTensor`): index tensor
            size = (batch, seq_len) or (batch, )
        Return:
            embeds: `FloatTensor`,
            size = (batch, seq_len, dim_embed) or (batch, dim_embed)
        """
        embeds = self.we(word_idxs)
        if self.position_encoding:
            embeds = self.pe(embeds, step=step)
        return embeds