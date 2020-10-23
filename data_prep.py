import numpy as np

# Gets snippets from converted batches
def get_snippet(batch_list, start, size):
    result_list = []
    for b in batch_list:
        result_list.append(b[:, start:start + size])
    
    return result_list

# Converts batch to shape that is suited to the pytorch sequential format
def convert_batches(batch_list):
    result_list = []
    for b in batch_list:
        frame_split = b.reshape(b.shape[0], -1, 1)
        frame_split = np.moveaxis(frame_split, 1, 0)
        
        result_list.append(frame_split)
    return result_list

# Converts batch back to original format
def revert_batch(batch):
    reorg = np.moveaxis(batch, 0, 1)
    reorg = reorg.reshape((batch.shape[1], -1))
    return reorg

# Uses pytorch format
def scale_samples(batch_list):
    r_list = []
    for b in batch_list:
        new_mat = b.astype(np.float32)
        for i in range(b.shape[1]):
            new_mat[:,i,:] = new_mat[:,i,:] / np.max(np.abs(new_mat[:,i,:]))
        r_list.append(new_mat)

    return r_list

# Squash matrix to frame size (helper for get_training_data)
def squash_to_frame_size(big_mat, frame_size, block_size, remove_zeros=True):
    num_points = block_size * frame_size

    iterations = big_mat.shape[0] // num_points

    if iterations == 0:
        raise "Error frame_size * block_size > time series"

    frame_entries = []

    for r in range(big_mat.shape[1]):
        for i in range(iterations):
            frame_row = big_mat[i * num_points: (i+1) * num_points, r, :].reshape(-1, block_size)
            if not remove_zeros or np.mean(np.abs(frame_row)) > 5.0:
                frame_entries.append(frame_row)
                
    return np.stack(frame_entries, axis=1)

# helper for get_training_data
def shuffle_sample_order(big_mat):
    num_samples = big_mat.shape[1]
    order = np.arange(num_samples)
    np.random.shuffle(order)
    return big_mat[:,order,:]

# returns list of pytorch sequential format np.ndarrays
# @frame_size number of timesteps
# @block_size number of pint per timestep
# @num_batches number of batches
# @batchsize number of samples in batch
def get_training_data(batch_list, frame_size, batch_size, num_batches, block_size=1):
    big_array = np.copy(batch_list[0])

    for batch in batch_list[1:]:
        big_array = np.concatenate((big_array, batch), axis = 0)
    
    squashed_data = squash_to_frame_size(big_array, frame_size, block_size)
    print(np.mean(np.abs(squashed_data[:,0,:])))
    shuffled_squashed_data = shuffle_sample_order(squashed_data)
    print(np.mean(np.abs(shuffled_squashed_data[:,0,:])))

    total_samples = batch_size * num_batches

    if total_samples > shuffled_squashed_data.shape[1]:
        raise "You requested more samples than I can offer!"

    batches = []

    for b in range(num_batches):
        batches.append(shuffled_squashed_data[:, b*batch_size:(b+1)*batch_size, :])

    return batches
