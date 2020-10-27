import librosa
import librosa.filters
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile

def load_wav(wav_path, sr=16000):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio


def save_wav(wav, path, hparams, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, hparams.sample_rate)


def trim_silence(wav, hparams):
    # These params are separate and tunable per dataset.
    unused_trimed, index = librosa.effects.trim(
        wav,
        top_db=hparams.trim_top_db,
        frame_length=hparams.
        trim_fft_size,
        hop_length=hparams.
        trim_hop_size)

    num_sil_samples = int(
        hparams.num_silent_frames * hparams.hop_size)
    start_idx = max(index[0] - int(num_sil_samples//2), 0)
    stop_idx = min(index[1] + num_sil_samples*2, len(wav))
    trimmed = wav[start_idx:stop_idx]
    return trimmed


def get_hop_size(hparams):
    return hparams.hop_size

def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db

    return _normalize(S, hparams)


def inv_mel_spectrogram(mel_spectrogram, hparams):
    D = _denormalize(mel_spectrogram, hparams)

    # Convert back to linear
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)

    return _griffin_lim(S ** hparams.power, hparams)


def _griffin_lim(S, hparams):
    '''
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=get_hop_size(hparams),
                        win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y,
                         hop_length=get_hop_size(hparams),
                         win_length=hparams.win_size)


_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate,
                               hparams.n_fft,
                               n_mels=hparams.acoustic_dim,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_db) /
                                                  (-hparams.min_db)) -
                   hparams.max_abs_value,
                   -hparams.max_abs_value, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_db >= 0
    return ((2 * hparams.max_abs_value) *
            ((S - hparams.min_db) / (-hparams.min_db)) -
            hparams.max_abs_value)


def _denormalize(D, hparams):
    return (((np.clip(D, -hparams.max_abs_value,
                      hparams.max_abs_value) + hparams.max_abs_value)
             * -hparams.min_db / (2 * hparams.max_abs_value))
             + hparams.min_db)

    return (((D + hparams.max_abs_value) * -hparams.min_db /
             (2 * hparams.max_abs_value)) + hparams.min_db)


