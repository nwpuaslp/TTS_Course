
import numpy as np
import librosa
from scipy import signal
import copy
import sys


def get_spectrograms(fpath, hparams):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hparams['sr'])

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=hparams['top_db'])

    # Preemphasis
    y = np.append(y[0], y[1:] - hparams['preemphasis'] * y[:-1])
    
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hparams['n_fft'],
                          hop_length=hparams['hop_length'],
                          win_length=hparams['win_length'])

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, hparams['n_fft'], hparams['n_mels'])  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hparams['ref_db'] + hparams['max_db']) / hparams['max_db'], 1e-8, 1)
    mag = np.clip((mag - hparams['ref_db'] + hparams['max_db']) / hparams['max_db'], 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def melspectrogram2wav(mel, hparams):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * hparams['max_db']) - hparams['max_db'] + hparams['ref_db']

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(hparams['sr'], hparams['n_fft'], hparams['n_mels'])
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hparams['preemphasis']], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hparams['max_db']) - hparams['max_db'] + hparams['ref_db']

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hparams['preemphasis']], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hparams['n_iter']):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hparams['n_fft'], hparams['hop_length'], win_length=hparams['win_length'])
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hparams['hop_length'], win_length=hparams['win_length'], window="hann")


if __name__ == "__main__":

    fpath = sys.argv[1]

    hparams={}
    hparams['sr'] = librosa.get_samplerate(fpath)
    hparams['n_fft'] = 512 # fft points (samples)
    hparams['frame_shift'] = 0.0025 # seconds
    hparams['frame_length'] = 0.01 # seconds
    hparams['hop_length'] = int(hparams['sr'] * hparams['frame_shift']) # samples.
    hparams['win_length'] = int(hparams['sr'] * hparams['frame_length']) # samples.
    hparams['n_mels'] = 80 # Number of Mel banks to generate
    hparams['power'] = 1.2 # Exponent for amplifying the predicted magnitude
    hparams['n_iter'] = 100 # Number of inversion iterations
    hparams['preemphasis'] = .97 # or None
    hparams['max_db'] = 100
    hparams['ref_db'] = 20
    hparams['top_db'] = 15

    mel, mag = get_spectrograms(fpath, hparams)

    wav = melspectrogram2wav(mel, hparams)

    sr = librosa.get_samplerate(fpath)
    librosa.output.write_wav("copy_synthesize/syn_mel.wav", wav, sr, norm=False)

