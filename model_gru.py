import torch.nn as nn
import numpy as np
import torch
import torch

class SimpleGRUEncoder(nn.Module):
    
    def __init__(self, frame_size, hidden_units):
        super(SimpleGRUEncoder, self).__init__()
        
        self.frame_size = frame_size
        self.hidden_units = hidden_units
        
        self.encoder_gru = nn.GRU(frame_size, hidden_units, num_layers=1)
        self.encoder_linear = nn.Linear(hidden_units, frame_size)

    def forward(self, data):
        hidden_state = self.encode(data)
            
        return hidden_state

    def encode(self, data):
        encoder_lstm_batch = data
        
        # encoder
        encoder_hidden_states = torch.randn(1, data.shape[1], self.hidden_units).cuda()
                
        _, encoder_hidden_states = self.encoder_gru(data, encoder_hidden_states)
        
        return encoder_hidden_states
    
    

    
class SimpleGRUDecoder(nn.Module):
    
    def __init__(self, frame_size, hidden_units, decode_hu=20):
        super(SimpleGRUDecoder, self).__init__()
        self.frame_size = frame_size
        self.hidden_units = hidden_units
                
        self.decoder_gru = nn.GRU(frame_size, hidden_units, num_layers=1)
        self.decoder_linear = nn.Linear(hidden_units, decode_hu)
        self.decoder_linear2 = nn.Linear(decode_hu, frame_size)
        self.activation_1 = torch.nn.Tanh()

    def forward(self, hidden_state, steps):
        return self.decode(hidden_state, steps)


    def decode(self, encoded_states, num_steps):
        
        decoder_input = torch.zeros(1, encoded_states.shape[1], self.frame_size).cuda()
        
        results = None
        
        for i in range(num_steps):
            decoder_out, encoded_states = self.decoder_gru(decoder_input, encoded_states)
            prediction = self.transform_output(decoder_out)
            decoder_input = prediction
            
            if results is None:
                results = prediction
            else:
                results = torch.cat((results, prediction), 0)
        
        return results

    def transform_output(self, input_value):
        val = self.decoder_linear(input_value)
        val = self.activation_1(val)
        val = self.decoder_linear2(val)

        return val