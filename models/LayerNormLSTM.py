import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNormLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        self.ln_i2h = torch.nn.LayerNorm(4 * hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(4 * hidden_size, elementwise_affine=False)
        
        self.ln_cell = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_hidden = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, state):
        h, c = state
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        i2h = self.i2h(x)  
        h2h = self.h2h(h) 

        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        ingate, forgetgate, cellgate, outgate = torch.split(preact, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)       
        forgetgate = torch.sigmoid(forgetgate) 
        cellgate = torch.tanh(cellgate)      
        outgate = torch.sigmoid(outgate)     

        c_t = forgetgate * c + ingate * cellgate
        c_t = self.ln_cell(c_t)  

        h_t = outgate * torch.tanh(c_t)
        h_t = self.ln_hidden(h_t) 

        return h_t, c_t


class LayerNormLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bias=True):
        super(LayerNormLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.layers = nn.ModuleList([
            LayerNormLSTMCell(
                input_size=(input_dim if layer == 0 else hidden_dim),
                hidden_size=hidden_dim,
                bias=bias
            ) for layer in range(num_layers)
        ])

    def forward(self, input: torch.Tensor, seq_lens=None, hidden_state=None):
        seq_len, batch_size, _ = input.size()  
        

        if hidden_state is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_dim, requires_grad=False)
            cx = input.new_zeros(self.num_layers, batch_size, self.hidden_dim, requires_grad=False)
        else:
            hx, cx = hidden_state


        outputs = []

        if seq_lens is not None:
            seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_dim, requires_grad=False)
            for i, l in enumerate(seq_lens):
                if l < seq_len:
                    seq_len_mask[i, l:, :] = 0  
            seq_len_mask = seq_len_mask.transpose(0, 1)  
        else:
            seq_len_mask = input.new_ones(seq_len, batch_size, self.hidden_dim, requires_grad=False)


        last_valid_indices = None
        if seq_lens is not None:

            if isinstance(seq_lens, torch.Tensor):
                lens_tensor = seq_lens
            else:
                lens_tensor = torch.tensor(seq_lens, device=input.device)
            last_valid_indices = (lens_tensor - 1).clamp(min=0)


        for t in range(seq_len):
            x = input[t] 
            layer_h = []
            layer_c = []
            

            for l, layer in enumerate(self.layers):
              
                h_prev = hx[l]
                c_prev = cx[l]
                

                h_t, c_t = layer(x, (h_prev, c_prev))
                

                h_t = h_t * seq_len_mask[t]
                c_t = c_t * seq_len_mask[t]
                
                layer_h.append(h_t)
                layer_c.append(c_t)

                x = h_t


            hx = torch.stack(layer_h)
            cx = torch.stack(layer_c)
            outputs.append(hx[-1])  


        output_sequence = torch.stack(outputs) 
        

        if seq_lens is not None and last_valid_indices is not None:

            final_h = []
            final_c = []
            for b in range(batch_size):
                t = last_valid_indices[b]
                final_h.append(hx[:, b, :][:, None, :])  
                final_c.append(cx[:, b, :][:, None, :])  
            final_hidden = torch.cat(final_h, dim=1)   
            final_cell = torch.cat(final_c, dim=1)      
        else:
            final_hidden = hx
            final_cell = cx

        return output_sequence, (final_hidden, final_cell)
