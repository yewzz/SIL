import torch
import torch.nn as nn
import math

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res



class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)


    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, x2, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        output2 = self.layer_norm1(x2)
        output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output2))
        value = self.transpose_for_scores(self.value(output2))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.1):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)
        #self.attention_block2 = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x,  pos=True, mask=None):
        features = x
        if pos:
            features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        features = self.attention_block(features,features, mask=mask)  # (batch_size, seq_len, dim)

        #features_cross = self.attention_block2()
        return features


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss