import numpy as np

def get_rfft_rvec(signal, hf_amp=30):
  fs = np.fft.rfft(signal, norm='ortho')

  fs *= np.logspace(0,1, num=fs.shape[0], base=hf_amp)

  vec = np.concatenate((fs.real, np.flip(fs.imag, 0)))

  return vec


def signal_to_ffts(signal, step_size):
  out = []
  for i in range(signal.shape[0] // step_size):
    fs = get_rfft_rvec(signal[i*step_size : (i+1)*step_size])
    out.append(fs)

  return np.array(out)


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

