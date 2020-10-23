import os

# Set google cloud credentials file path to credentials directory
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getcwd() + "/credentials/key.json"

import data_loader as dl
import data_prep as dp

# Load training data
batches = dl.load_dataset("training_data_3", "Batch", 1)

def get_data(frame_size, batch_size, num_batches, block_size=1):
    converted_batches = dp.convert_batches(batches)
    batch_list = dp.get_training_data(converted_batches, frame_size, batch_size, num_batches, block_size)
    scaled_batches = dp.scale_samples(batch_list)
    return scaled_batches
