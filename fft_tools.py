import numpy as np

def get_rfft_rvec(signal, hf_amp=30):
  fs = np.fft.rfft(signal, norm='ortho')

  fs *= np.logspace(0,1, num=fs.shape[0], base=hf_amp)

  vec = np.concatenate((fs.real, np.flip(fs.imag, 0)))

  return vec


def signal_to_ffts(signal, step_size, hf_amp=30):
  out = []
  for i in range(signal.shape[0] // step_size):
    fs = get_rfft_rvec(signal[i*step_size : (i+1)*step_size], hf_amp=hf_amp)
    out.append(fs)

  return np.array(out)


def signal_batch_to_ffts(signal_batch, step_size, hf_amp=30):
    tsl = []
    for b in range(signal_batch.shape[1]):
        s = signal_batch[:, b, :].reshape((-1,))
        ts = signal_to_ffts(s, step_size, hf_amp=hf_amp)
        tsl.append(ts)
    
    return np.stack(tsl, axis=1)


def dataset_to_ffts(dataset, step_size, hf_amp=30):
    result_set = []
    for d in dataset:
        tb = signal_batch_to_ffts(d, step_size, hf_amp=hf_amp)
        result_set.append(tb)
    return result_set


def invert_fft_batch(fft_batch, hf_damp=30):
    ifl = []
    for b in range(fft_batch.shape[1]):
        inv_s = invert_ffts(fft_batch[:, b, :], hf_damp=hf_damp)
        ifl.append(inv_s)
    
    return np.stack(ifl, axis=0)


def invert_ffts(fft_mat, hf_damp=30):
  damper = np.logspace(0,1, num=fft_mat.shape[1] // 2, base=hf_damp)

  out = []

  for i in range(fft_mat.shape[0]):
    fs_ser = fft_mat[i]
    fs_real = fs_ser[:fs_ser.shape[0]//2]
    fs_imag = np.flip(fs_ser[fs_ser.shape[0]//2:])

    fs = fs_real + fs_imag * 1j

    fs /= damper

    fs_signal = np.fft.irfft(fs, norm='ortho')
    out.append(fs_signal)

  return np.concatenate(out)

