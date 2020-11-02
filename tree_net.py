import torch
from model_gru import SimpleGRUEncoder
from model_gru import SimpleGRUDecoder
import torch.optim as optim
import helper

class TreeNet:

    # Builds up a tree
    # sizes of decoder and encoder nets are equal
    def __init__(self, hidden_units, frame_size, time_steps_level, lr, pretrained=[]):
        self.enocders = []
        self.decoders = []

        self.time_steps_level = time_steps_level
        self.hidden_units = hidden_units
        self.frame_size = frame_size

        # GPU device
        device = torch.device("cuda")

        parameter_list = []

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

                parameter_list += enc.parameters()
                parameter_list += dec.parameters()

            self.enocders.append(enc)
            self.decoders.append(dec)

        print("Parameter list contains " + str(len(parameter_list)) + " entries!")
        
        if len(parameter_list) > 0:
            self.optimizer = optim.Adam(parameter_list, lr=lr)
        else:
            self.optimizer = None
        
    def forward(self, input_d):
        if input_d.shape[0] != self.get_total_datablocks():
            raise RuntimeError("The shape of the first dimension of the input data must equal the number of blocks required by the network. Got " + str(input_d[0].shape[0]) + " and " + str(self.get_total_datablocks()))

        return self.get_forward_for_level(input_d)

    def get_forward_for_level(self, input_data, running_index=0, level=None):
        if level is None:
            level = len(self.hidden_units) - 1
        
        if level > 0:
            data_list = []
            for i in range(self.time_steps_level[level]):
                data_list.append(self.get_forward_for_level(input_data, level= level - 1, running_index= running_index + i * self.get_db_for_ts_in_lvl(level)))

            # Concat to time series tensor
            data = torch.cat(data_list, dim=0)

            return self.enocders[level](data)
        else:
            return self.enocders[0](input_data[running_index : running_index + self.time_steps_level[0]])
            
    def backward(self, state_data, level=None):
        if level is None:
            level = len(self.time_steps_level) - 1
        result_list = []

        predictions = self.decoders[level](state_data, self.time_steps_level[level])

        if level > 0:
            for i in range(self.time_steps_level[level]):
                result_list.append(self.backward(predictions[i:i+1], level = level - 1))

            return torch.cat(result_list, dim=0)
        else:
            return predictions
            

    # Gets the amount of datablocks for one timestep in the dedicated level (use num_levels + 1 for overall datablocks amount)
    def get_db_for_ts_in_lvl(self, level):
        result = 1
        for i in self.time_steps_level[:level]:
            result *= i
        return result

    def get_total_datablocks(self):
        return self.get_db_for_ts_in_lvl(len(self.time_steps_level))

    def zero_grad(self):
        for e in self.enocders + self.decoders:
            e.zero_grad()

    def save(filename):
        for i, e in enumerate(self.encoders):
            torch.save(e, filename + "-encoder-l" + str(i)+".mdl")
            
        for i, d in enumerate(self.decoders):
            torch.save(d, filename + "-decoder-l" + str(i)+".mdl")
            
