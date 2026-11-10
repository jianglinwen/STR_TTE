import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from models.LayerNormLSTM import LayerNormLSTM
from transformers import BertConfig, BertForMaskedLM

batch_first=False
bidirectional=False

class STR_TTE(nn.Module):
    def __init__(self, input_dim, seq_input_dim, seq_hidden_dim, seq_layer, bert_hiden_size, pad_token_id,
                 bert_attention_heads, bert_hidden_layers, decoder_layer, decode_head, vocab_size=27300):

        super(STR_TTE, self).__init__()
        self.bert_config = BertConfig(num_attention_heads = bert_attention_heads, hidden_size = bert_hiden_size, pad_token_id=pad_token_id,
                                      vocab_size=vocab_size, num_hidden_layers = bert_hidden_layers)
        self.seg_embedding_learning = BertForMaskedLM(self.bert_config) 

        self.vehembed = nn.Embedding(17, 5, padding_idx=0)  
        self.cardidembed = nn.Embedding(353343, 8)
        self.weekembed = nn.Embedding(8, 3)                              
        self.dateembed = nn.Embedding(32, 10)
        self.timeembed = nn.Embedding(1441, 20)               
        self.gpsrep = nn.Linear(4, 16)                        
        self.timene_dim = 3 + 10 + 20 + bert_hiden_size

        self.timene = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, self.timene_dim)
        )

        self.represent = nn.Sequential(
            nn.Linear(input_dim, seq_input_dim), 
            nn.LeakyReLU(),
            nn.Linear(seq_input_dim, seq_input_dim)
        )


        self.sequence = LayerNormLSTM(seq_input_dim, seq_hidden_dim, seq_layer) 
        self.seq_hidden_dim = seq_hidden_dim * 2 if bidirectional else seq_hidden_dim


        feed_forward_dim=128
        num_heads = 1
        num_layers = 3
        dropout = 0.1
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(seq_input_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.decoder_embed_dim = seq_hidden_dim * 2 if bidirectional else seq_hidden_dim


        self.input2hid = nn.Linear(seq_hidden_dim+33, seq_hidden_dim)


        self.decoder = Decoder(d_model=self.decoder_embed_dim, N=decoder_layer, heads=decode_head)

        self.hid2out = nn.Linear(self.seq_hidden_dim, 1) 


    def pooling_sum(self, hiddens, lens):
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False) 
        batch_size = range(hiddens.shape[0]) 
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]

    def seg_embedding(self, x):
        bert_output = self.seg_embedding_learning(input_ids=x[0], encoder_attention_mask=x[1],  labels=x[2], output_hidden_states=True)

        return bert_output["loss"], bert_output["hidden_states"][4], bert_output["logits"]

    def forward(self, inputs, args):
        feature = inputs['links']

        lens = inputs['lens']
        vehrep = self.vehembed(feature[:, :, 0].long()) 
        weekrep = self.weekembed(feature[:, :, 3].long()) 
        daterep = self.dateembed(feature[:, :, 4].long()) 
        timerep = self.timeembed(feature[:, :, 5].long())
        gpsrep = self.gpsrep(feature[:, :, 6:10]) 


        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) 

        loss_1, hidden_states, prediction_scores = self.seg_embedding([inputs['linkindex'], inputs['encoder_attention_mask'], inputs['mask_label']]) 

        timene_input = torch.cat([self.seg_embedding_learning.bert.embeddings.word_embeddings(inputs['rawlinks']), datetimerep], dim=-1) 
        timene = self.timene(timene_input)+timene_input 
        representation = self.represent(torch.cat([feature[..., 1:3], vehrep, gpsrep, timene], dim=-1))  

        representation = representation if batch_first else representation.transpose(0, 1).contiguous()
        hiddens, rnn_states = self.sequence(representation, seq_lens=lens.long())

        decoder = self.decoder(hiddens, lens) 

        decoder = decoder if batch_first else decoder.transpose(0, 1).contiguous() 
        pooled_decoder = self.pooling_sum(decoder, lens) 
        pooled_hidden = torch.cat([pooled_decoder, weekrep[:, 0], daterep[:, 0], timerep[:, 0]], dim=-1)
        hidden = F.leaky_relu(self.input2hid(pooled_hidden)) 
        output = self.hid2out(hidden) 
        output = args.scaler.inverse_transform(output) 
        return output, loss_1


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model 

        self.alpha = nn.Parameter(torch.ones(self.size)) 
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads 
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model) 
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, len):

        k = self.k_linear(k) 
        q = self.q_linear(q) 
        v = self.v_linear(v)
        S = q.shape[0] 
        mask = torch.stack([torch.cat((torch.zeros(i), torch.ones(S - i)), 0) for i in len]).bool().to(k.device)
        attn_output, attn_output_weights = self.attn_1(q, k, v, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()


        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model) 
        self.norm_2 = Norm(d_model) 
        self.norm_3 = Norm(d_model) 

        self.dropout_1 = nn.Dropout(dropout) 
        self.dropout_2 = nn.Dropout(dropout) 
        self.dropout_3 = nn.Dropout(dropout) 

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout) 
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout) 
        self.ff = FeedForward(d_model, dropout=dropout) 


    def forward(self, x, len):
        x2 = self.norm_2(x) 
        x = x + self.dropout_2(self.attn_2(x2, x2, x2, len))
        x2 = self.norm_3(x) 
        x = x + self.dropout_3(self.ff(x2)) 
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) 

class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N 
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N) 
        self.norm = Norm(d_model) 

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)
    
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask):

        batch_size = query.shape[0] 
        tgt_length = query.shape[-2] 
        src_length = key.shape[-2] 

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        ) 

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  
        if mask is not None:

            valid_lens = mask.sum(dim=1, dtype=torch.long)  
            batch_size, seq_len = mask.shape
            target_dim1 = seq_len 


            result = torch.zeros(batch_size, target_dim1, seq_len, device=mask.device, dtype=mask.dtype)

            for i in range(batch_size):
                vl = valid_lens[i]  

                result[i, :vl, :] = mask[i].unsqueeze(0).repeat(vl, 1)
              
            mask = result.unsqueeze(1)
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask, dim=-2):
        x = x.transpose(dim, -2) 

        residual = x
        out = self.attn(x, x, x, mask)  
        out = self.dropout1(out) 
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2) 
        return out


