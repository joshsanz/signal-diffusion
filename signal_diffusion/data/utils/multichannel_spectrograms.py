# Multichannel Spectrogram Utilities
import librosa
import numpy as np
from PIL import Image
from functools import lru_cache


# Cache DFT matrices to avoid recomputing
@lru_cache(maxsize=32)
def lin_dftmtx(N):
    is_odd = N % 2 == 1
    spacing = 1 / N
    freqs = np.concatenate([
        np.linspace(-0.5, 0, N // 2 + 1, endpoint=True),
        np.linspace(spacing, 0.5, (N - 1) // 2, endpoint=is_odd)
    ])
    omega = np.exp(-2j * np.pi / N)
    W = omega ** (np.outer(np.fft.ifftshift(freqs) * N, np.arange(N))) / np.sqrt(N)
    return freqs, W


# Cache DFT matrices to avoid recomputing
@lru_cache(maxsize=32)
def log_dftmtx(N, min_exponent=-3):
    is_odd = N % 2 == 1
    neg_freqs = [-(10 ** i) for i in np.linspace(np.log10(0.5), min_exponent, N // 2, endpoint=True)]
    freqs = np.array(neg_freqs + [0] + [-f for f in neg_freqs[not is_odd:][::-1]])
    omega = np.exp(-2j * np.pi / N)
    W = omega ** (np.outer(np.fft.ifftshift(freqs) * N, np.arange(N))) / np.sqrt(N)
    return freqs, W


def fft2rfft(X):
    N = X.shape[0]
    X = X.reshape(N, -1)
    is_odd = N % 2 == 1
    is_even = N % 2 == 0
    split_idx = N // 2 + is_odd
    rX = X[:split_idx, :]
    rX[1:, :] += X[split_idx + is_even:, :][::-1, :].conjugate()
    return rX.squeeze()


def apply_stft(x, W, win_length, hop_length, window=None, pad=True, demean=True):
    if demean:
        # Remove DC component during DFT to avoid large sidelobes
        # Insert DC component back in the first row at end
        dc = x.sum() / np.sqrt(x.shape[0])
        x = x - x.mean()
    # Zero-pad to fit integer number of frames
    if pad:
        padding = win_length - (x.shape[0] % hop_length)
        x = np.concatenate([x, np.zeros(padding, dtype=x.dtype)])
    if window == 'hann':
        window = np.hanning(win_length)
    elif isinstance(window, np.ndarray):
        assert window.shape == (win_length,), "Window must be of shape (win_length,)"
    elif window is None:
        window = np.ones(win_length)
    # Form strided window view of x and apply DFT
    x_strided = np.lib.stride_tricks.sliding_window_view(x, win_length)[::hop_length]
    X = W @ (x_strided.T * window.reshape(win_length, 1))
    if demean:
        X[0, :] = dc
    return X


def apply_rstft(x, W, win_length, hop_length, window=None, pad=True, demean=True):
    assert x.dtype == np.float32 or x.dtype == np.float64, "Only real-valued x is allowed"
    X = apply_stft(x, W, win_length, hop_length, window, pad)
    rX = fft2rfft(X)
    return rX


def log_rstft(x, win_length, hop_length, window=None, pad=True, demean=True):
    f, W = log_dftmtx(win_length)
    return f, apply_rstft(x, W, win_length, hop_length, window)


def lin_rstft(x, win_length, hop_length, window=None, pad=True, demean=True):
    f, W = lin_dftmtx(win_length)
    return f, apply_rstft(x, W, win_length, hop_length, window)


def scale_minmax(X, minval=0.0, maxval=1.0):
    xmax = X.max()
    xmin = X.min()
    if xmax == xmin:
        X_std = np.zeros_like(X)
    else:
        X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (maxval - minval) + minval
    return X_scaled


def scale_to_uint8(X, data_min: float, data_max: float) -> np.ndarray:
    """Clip ``X`` to ``[data_min, data_max]`` and map to uint8."""
    if data_max <= data_min:
        return np.zeros_like(X, dtype=np.uint8)
    clipped = np.clip(X, data_min, data_max)
    scaled = (clipped - data_min) / (data_max - data_min)
    return (scaled * 255).astype(np.uint8)


def spectrogram(
    y,
    hop_length,
    win_length,
    noise_floor_db=-130.,
    bin_spacing='linear',
    output_type='db-only',
    min_db=None,
    max_db=None,
):
    """Generate spectrogram with configurable output format.

    Parameters:
    - y: Input signal
    - hop_length: STFT hop length
    - win_length: STFT window length
    - noise_floor_db: Noise floor for dB clipping (default: -130)
    - bin_spacing: 'linear' or 'log' frequency spacing
    - min_db: Optional lower bound (dB) for clipping and scaling (defaults to noise_floor_db)
    - max_db: Optional upper bound (dB) for clipping and scaling (defaults to observed max)
    - output_type: Output format - 'db-only', 'db-iq', or 'db-polar'

    Returns:
    - For 'db-only': (height, width) grayscale uint8 array
    - For 'db-iq': (height, width, 3) RGB uint8 array [dB, I, Q]
    - For 'db-polar': (height, width, 3) RGB uint8 array [dB, magnitude, phase]
    """
    if bin_spacing == 'linear':
        stft = librosa.stft(
            y=y, win_length=win_length, hop_length=hop_length, n_fft=win_length * 2 - 1,
        )
    elif bin_spacing == 'log':
        _, stft = log_rstft(
            x=y, win_length=win_length * 2 - 1, hop_length=hop_length, window='hann',
        )
    else:
        raise ValueError(f"Invalid bin spacing {bin_spacing}")

    # Compute dB magnitude (channel 0 for all modes)
    stft_db = 20 * np.log10(np.abs(stft) + 1e-9)
    clip_min_db = noise_floor_db if min_db is None else min_db
    clip_max_db = max_db
    if clip_max_db is not None and clip_max_db <= clip_min_db:
        raise ValueError("max_db must be greater than min_db when both are provided")
    if clip_max_db is None:
        stft_db = np.maximum(stft_db, clip_min_db)
        scale_max_db = stft_db.max()
    else:
        stft_db = np.clip(stft_db, clip_min_db, clip_max_db)
        scale_max_db = clip_max_db
    db_channel = scale_to_uint8(stft_db, clip_min_db, scale_max_db)
    db_channel = np.flip(db_channel, axis=0)  # put low frequencies at the bottom

    if output_type == 'db-only':
        return db_channel

    elif output_type == 'db-iq':
        # Extract real (I) and imaginary (Q) components
        i_channel = np.real(stft)
        q_channel = np.imag(stft)

        # Scale I and Q identically using global min/max to preserve their relationship
        global_min = min(i_channel.min(), q_channel.min())
        global_max = max(i_channel.max(), q_channel.max())

        if global_max == global_min:
            i_scaled = np.zeros_like(i_channel)
            q_scaled = np.zeros_like(q_channel)
        else:
            i_scaled = (i_channel - global_min) / (global_max - global_min)
            q_scaled = (q_channel - global_min) / (global_max - global_min)

        i_channel = (i_scaled * 255).astype(np.uint8)
        q_channel = (q_scaled * 255).astype(np.uint8)

        # Flip to match dB orientation
        i_channel = np.flip(i_channel, axis=0)
        q_channel = np.flip(q_channel, axis=0)

        # Stack as (height, width, 3)
        img = np.stack([db_channel, i_channel, q_channel], axis=-1)
        return img

    elif output_type == 'db-polar':
        # Linear magnitude (not dB)
        magnitude = np.abs(stft)
        magnitude_channel = scale_minmax(magnitude, 0, 255).astype(np.uint8)
        magnitude_channel = np.flip(magnitude_channel, axis=0)

        # Phase in [-π, π)
        phase = np.angle(stft)  # Returns phase in [-π, π]
        # Map [-π, π] to [0, 255]
        phase_channel = scale_minmax(phase, 0, 255).astype(np.uint8)
        phase_channel = np.flip(phase_channel, axis=0)

        # Stack as (height, width, 3)
        img = np.stack([db_channel, magnitude_channel, phase_channel], axis=-1)
        return img

    else:
        raise ValueError(f"Invalid output_type '{output_type}'. Must be 'db-only', 'db-iq', or 'db-polar'")


def multichannel_spectrogram(
    x,
    resolution,
    hop_length,
    win_length,
    noise_floor_db=-130.,
    bin_spacing='linear',
    output_type='db-only',
    min_db=None,
    max_db=None,
):
    """
    Create a multichannel spectrogram image from a 2D array X

    # Parameters:
    - x (np.ndarray): channels x samples array of real values
    - resolution (int): width and height of the square output image
    - hop_length (int): STFT hop length
    - win_length (int): STFT window length; n_fft will be win_length * 2 - 1 to
                        enforce win_length output samples
    - noise_floor_db (float): noise floor in dB to clip the STFT output to, helps image dynamic range
    - bin_spacing (str): 'linear' or 'log', spacing of frequency bins in the STFT
    - output_type (str): output format - 'db-only', 'db-iq', or 'db-polar'
    - min_db (float | None): lower bound for clipping/scaling (defaults to noise_floor_db)
    - max_db (float | None): upper bound for clipping/scaling (defaults to observed max)
    """
    specs = [
        spectrogram(
            x[i, :],
            hop_length,
            win_length,
            noise_floor_db,
            bin_spacing,
            output_type,
            min_db,
            max_db,
        )
        for i in range(x.shape[0])
    ]
    swidth = specs[0].shape[1]
    Width = swidth * len(specs)
    assert Width <= resolution, f"Image width {Width}={len(specs)}*{swidth} is greater than resolution {resolution}"
    width_pad = (resolution - Width) // 2

    # Determine image mode based on output_type
    if output_type == 'db-only':
        mode = 'L'
    else:  # 'db-iq' or 'db-polar'
        mode = 'RGB'

    merged = Image.new(mode, (resolution, resolution))
    for i, spec in enumerate(specs):
        merged.paste(Image.fromarray(spec, mode=mode), (width_pad + i * swidth, 0))
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
