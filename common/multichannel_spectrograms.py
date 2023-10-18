# Multichannel Spectrogram Utilities
import librosa
import numpy as np
from PIL import Image


def scale_minmax(X, minval=0.0, maxval=1.0):
    xmax = X.max()
    xmin = X.min()
    if xmax == xmin:
        X_std = np.zeros_like(X)
    else:
        X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (maxval - minval) + minval
    return X_scaled


def spectrogram(y, hop_length, win_length, noise_floor_db=-100, bin_spacing='linear'):
    if bin_spacing == 'linear':
        stft = librosa.stft(
            y=y, win_length=win_length, hop_length=hop_length, n_fft=win_length * 2 - 1,
        )
    elif bin_spacing == 'log':
        lin_freqs = np.linspace(-0.5, 0.5, win_length, endpoint=True)
        log_freqs = ([-(10 ** i) for i in np.linspace(0, -4, win_length // 2, endpoint=True)] + [0] +
                     [10 ** i for i in np.linspace(-4, 0, (win_length - 1) // 2, endpoint=True)])

    else:
        raise ValueError(f"Invalid bin spacing {bin_spacing}")
    stft = 20 * np.log10(np.abs(stft) + 1e-9)

    # Set noise floor to reduce dynamic range
    stft = np.clip(stft, noise_floor_db, None)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(stft, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    # img = 255 - img  # invert. make black==more energy
    return img


def multichannel_spectrogram(x, resolution, hop_length, win_length, noise_floor_db=-100):
    """
    Create a multichannel spectrogram image from a 2D array X

    # Parameters:
    - x (np.ndarray): channels x samples array of real values
    - resolution (int): width and height of the square output image
    - hop_length (int): STFT hop length
    - win_length (int): STFT window length; n_fft will be win_length * 2 - 1 to
                        enforce win_length output samples
    - noise_floor_db (float): noise floor in dB to clip the STFT output to, helps image dynamic range
    """
    specs = [spectrogram(x[i, :], hop_length, win_length, noise_floor_db) for i in range(x.shape[0])]
    swidth = specs[0].shape[1]
    Width = swidth * len(specs)
    assert Width <= resolution, f"Image width {Width}={len(specs)}*{swidth} is greater than resolution {resolution}"
    width_pad = (resolution - Width) // 2
    merged = Image.new("L", (resolution, resolution))
    for i, spec in enumerate(specs):
        merged.paste(Image.fromarray(spec, mode='L'), (width_pad + i * swidth, 0))
    return merged


def multichannel_spectrogram_griffinlim(image, n_channels, n_samples, hop_length=128, win_length=None, is_db=True):
    """
    Using Griffin-Lim algorithm, reconstruct the multichannel time-domain signal from a multichannel
    magnitude spectrogram image.

    # Parameters:
    - image (PIL.Image): multichannel spectrogram image
    - n_channels (int): number of channels in the original signal
    - n_samples (int): number of samples in the original signal (used to determine channel boundaries)
    - hop_length (int): STFT hop length
    - win_length (Optional[int]): optional STFT window length, if None, will be set to image height
    - is_db (bool): if True, will convert dB to magnitude before reconstruction
    """
    if win_length is None:
        win_length = image.height
    # Reformat input image for librosa
    X = np.asarray(image)
    X = np.flip(X, axis=0)  # Convert from human-friendly order to librosa format
    if is_db:
        X = 10 ** (X / 20)
    # Normalize values to something reasonable
    X = scale_minmax(X, 0, 1)
    # Determine channel padding & locations
    chwidth = int(np.ceil(n_samples / hop_length))
    chpad = (X.shape[1] - n_channels * chwidth) // 2
    # Reconstruct each channel
    x = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        Xch = X[:, chpad + i * chwidth:chpad + (i + 1) * chwidth]
        # Add noise?
        Xch += np.random.randn(*Xch.shape) * 1e-3
        gl = librosa.griffinlim(Xch, hop_length=hop_length, win_length=win_length, n_fft=win_length * 2 - 1)
        x[i, :] = np.pad(gl, (0, n_samples - len(gl)), 'constant', constant_values=0)
    return x / np.linalg.norm(x)  # Normalize, since scaling was lost in original spectrogram conversion
