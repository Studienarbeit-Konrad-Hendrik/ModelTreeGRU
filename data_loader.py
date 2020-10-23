import bucket_handler as bh
import numpy as np
import io

# TODO: store the batchsize somewhere!!! eg. use another format not raw bytes...
def load_dataset(bucket_name, name, num_batches):
    bucket = bh.get_or_create_bucket(bucket_name)
    batch_list = []
    for i in range(num_batches):
        batch = np.load(io.BytesIO(bh.get_bytes_from_blob(bucket, "data/"+name+"_"+str(i)+".bin")))
        batch_list.append(batch)

    return batch_list
