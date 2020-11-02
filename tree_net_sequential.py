import torch
from model_gru import SimpleGRUEncoder
from model_gru import SimpleGRUDecoder
import torch.optim as optim
import helper

class TreeNet:

    # Builds up a tree
    # sizes of decoder and encoder nets are equal
    def __init__(self, hidden_units, frame_size, time_steps_level, lr, pretrained=[]):
        self.encoders = []
        self.decoders = []

        self.time_steps_level = time_steps_level
        self.hidden_units = hidden_units
        self.frame_size = frame_size

        # GPU device
        device = torch.device("cuda")

        self.optimizers = []

        # Add networks
        for i, t in enumerate(hidden_units):
            if helper.is_in_pretrained(pretrained, i):
                p = helper.get_pretrained_for(pretrained, i)
                enc = p['encoder']
                dec = p['decoder']
            else:
                if i == 0:
                    enc = SimpleGRUEncoder(frame_size, t).to(device)
                    dec = SimpleGRUDecoder(frame_size, t, decode_hu=((frame_size + t) // 2)).to(device)
                else:
                    enc = SimpleGRUEncoder(hidden_units[i - 1], t).to(device)
                    dec = SimpleGRUDecoder(hidden_units[i - 1], t, decode_hu=((frame_size + t) // 2)).to(device)

                opt_params = list(enc.parameters()) + list(dec.parameters())
                
                self.optimizers.append(optim.Adam(opt_params, lr=lr))

            self.encoders.append(enc)
            self.decoders.append(dec)        
        
    
    def get_data_for_level(self, data, level):
        if level == 0:
            return data
        else:
            features = []
            for c in range(int(self.get_data_for_level(data, level-1).shape[0] / self.time_steps_level[level-1])):
                d = self.forward(
                    self.get_data_for_level(data, level-1)[
                        c*self.time_steps_level[level-1] : (c+1)*self.time_steps_level[level-1]
                    ], level=level-1)
                features.append(d)
            return torch.cat(features, dim=0)

    def forward(self, input_data, level=0):
        return self.encoders[level](input_data)
                
    def backward(self, state_data, level=0):
        return self.decoders[level](state_data, self.time_steps_level[level])

    # Gets the amount of datablocks for one timestep in the dedicated level (use num_levels + 1 for overall datablocks amount)
    def get_db_for_ts_in_lvl(self, level):
        result = 1
        for i in self.time_steps_level[:level]:
            result *= i
        return result

    def get_total_datablocks(self):
        return self.get_db_for_ts_in_lvl(len(self.time_steps_level))

    def zero_grad(self):
        for e in self.encoders + self.decoders:
            e.zero_grad()

    def save(filename):
        for i, e in enumerate(self.encoders):
            torch.save(e, filename + "-encoder-l" + str(i)+".mdl")
            
        for i, d in enumerate(self.decoders):
            torch.save(d, filename + "-decoder-l" + str(i)+".mdl")
            
