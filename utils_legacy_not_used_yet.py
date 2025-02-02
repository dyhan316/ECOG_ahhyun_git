import os 
import pandas as pd
import numpy as np
from scipy import signal, stats
import torch 
from superlet import superlet #! added by 스ㅇ주쌤

def pd_train_val_test_split(labels_pd, valid_split, test_split, shuffle_seed=None):
    n = len(labels_pd)
    valid_n = int(n * valid_split)
    test_n = int(n * test_split)
    train_n = n - valid_n - test_n
    
    if shuffle_seed is not None:
        labels_pd = labels_pd.sample(frac=1, random_state=shuffle_seed)
    
    train_pd = labels_pd.iloc[:train_n]
    valid_pd = labels_pd.iloc[train_n:train_n + valid_n]
    test_pd = labels_pd.iloc[train_n + valid_n:]
    
    return train_pd, valid_pd, test_pd


def arr_from_pd(data_dir, data_pd):
    data = []
    labels = []
    for file in data_pd['Data File']:
        file_path = os.path.join(data_dir, file)
        data.append(np.load(file_path))
    labels = data_pd['Label'].values
    return np.array(data), labels

def make_ready_for_BrainBERT(data, device = 'cuda'):
    r""" 
    data : np array of shape (samples, num_channels, f_bin, time_len) #* this is the shape of the output of get_stft_multi_channel
    """
    assert data.ndim == 4, "data must have shape (samples, num_channels, f_bin, time_len)"
    assert data.shape[1] == 1, "only implemented for single channel! (actually could be used for multi-channel too, but not what BrainBERT was designed to do)"
    
    data = np.squeeze(data, axis=1)
    data = np.transpose(data, (0, 2, 1))

    data = torch.FloatTensor(data).to(device)
    return data
    
def revert_BrainBERT_back(data):
    r"""
    data : output of BrainBERT, would be shape of (samples, time_len, f_bin)
    returns : shape of (samples, f_bin, time_len) and convert back to numpy array
    """
    data = data.cpu().numpy()
    data = np.transpose(data, (0, 2, 1))
    return data    

def segment_data_and_labels(data, labels, segment_length=10, stride=None):
    """
    Reshape data by dividing the last dimension into segments and repeat labels accordingly.

    Parameters
    ----------
    data : np.ndarray
        The input data with shape (samples, num_channels, f_bin, time_len).
    labels : np.ndarray
        The labels associated with each sample in data.
    segment_length : int, optional
        The length of each segment after division. Default is 10.
    stride : int, optional
        The stride to use when segmenting the data. Default is None.

    Returns
    -------
    reshaped_data : np.ndarray
        The reshaped data with shape (samples * num_segments, num_channels, f_bin, segment_length).
    repeated_labels : np.ndarray
        The labels repeated for each segment.
    """
    assert data.ndim == 4, "data must have shape (samples, num_channels, f_bin, time_len)"
    if stride is None:
        stride = segment_length
        
    num_samples, num_channels, f_bin, time_len = data.shape
    num_segments = ((time_len - segment_length) // stride) + 1
    
    segments_list = []
    
    for start in range(0, time_len - segment_length + 1, stride):
        end = start + segment_length
        segment = data[:, :, :, start:end]
        segments_list.append(segment)
    
    reshaped_data = np.concatenate(segments_list, axis=0)
    
    # Adjust the repetition of labels to match the new number of segments
    repeated_labels = np.repeat(labels, num_segments)
    
    return reshaped_data, repeated_labels


def get_stft_multi_channel(x, fs, clip_fs=-1, normalizing=None, boundary_clip=5, batch_dim=False, **kwargs):
    """
    Compute the Short-Time Fourier Transform (STFT) for multi-channel signals.

    Parameters
    ----------
    x : ndarray
        Input signal of shape [channel, time_len] or [time_len] (in which case channel is added). (if batch_dim is True, [batch, channel, time_len])
    fs : int
        Sampling frequency of the input signal.
    clip_fs : int
        Number of frequency bins to keep (i.e., frequency clipping; clips higher frequencies).
    normalizing : str
        Type of normalization to apply to the STFT. Options are "zscore", "baselined", "db".
    boundary_clip : int
        Number of bins to clip from each end of the spectrum to handle boundary effects.
    batch_dim : bool, optional
        Whether the input signal has an additional batch dimension in front. Default is False.
    **kwargs : dict
        Additional keyword arguments to pass to scipy.signal.stft.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of x. Shape is [batch, channel, frequencies, times] if input has batch dimension.
        Shape is [channel, frequencies, times] if input does not have batch dimension.
    """
    # assert clip_fs > 0, "clip_fs must be greater than 0" #! commented out by 스ㅇ주쌤
    # Check if x is one-dimensional (single channel)
    if x.ndim == 1:
        x = x[np.newaxis, :]  # Make it two-dimensional for consistency

    # Check if x has batch dimension
    if batch_dim:
        assert x.ndim == 3, "x must have shape [batch, channel, time_len] when batch_dim is True"
        batch_size = x.shape[0]
    else:
        assert x.ndim == 2, "x must have shape [channel, time_len] when batch_dim is False"
        batch_size = 1

    # Initialize list to hold STFT results for each channel
    Zxx_list = []

    for batch in range(batch_size):
        Zxx_batch_list = []
        x_batch = x[batch] if batch_dim else x
        for channel in range(x_batch.shape[0]):
            f,t,Zxx_channel = get_stft(x_batch[channel], fs, clip_fs=clip_fs, normalizing=normalizing, boundary_clip=boundary_clip, **kwargs)
            Zxx_batch_list.append(Zxx_channel)
        # Stack the results for each channel along a new dimension
        Zxx_list.append(np.stack(Zxx_batch_list, axis=0))
    # Stack the results for each batch along a new dimension
    Zxx = np.stack(Zxx_list, axis=0)
    
    return f, t, Zxx

#! TODO : make this better organized and compartmentalized!

#*below version : not used since single channel version..
def get_stft(x, fs, clip_fs=-1, normalizing=None, boundary_clip = 5,**kwargs):
    """
    #write documentation 
    fs : int
        Sampling frequency of the input signal
    clip_fs : int
        Number of frequency bins to keep (i.e. frequency clipping (clips higher frequencies))
    normalizing : str
        Type of normalization to apply to the STFT. Options are "zscore", "baselined", "db"
    boundary_clip : int
        Number of time bins to clip from the beginning and end of the time axis (to handle boundary effects (artifects and so on))
    **kwargs : dict
        Additional keyword arguments to pass to scipy.signal    
    """
    f, t, Zxx = signal.stft(x, fs, **kwargs)
    
    Zxx = Zxx[:clip_fs]
    f = f[:clip_fs]

    Zxx = np.abs(Zxx)
    
    if normalizing=="zscore":
        Zxx = Zxx[:,boundary_clip:-boundary_clip]
        Zxx = stats.zscore(Zxx, axis=-1)
        t = t[boundary_clip:-boundary_clip]
    elif normalizing=="baselined":
        raise NotImplementedError
        # Zxx = baseline(Zxx[:,boundary_clip:-boundary_clip])
        # t = t[boundary_clip:-boundary_clip]
    elif normalizing=="db":
        Zxx = np.log2(Zxx[:,boundary_clip:-boundary_clip])
        t = t[boundary_clip:-boundary_clip]

    if np.isnan(Zxx).any():
        import pdb; pdb.set_trace()

    return f, t, Zxx

#! added by 스ㅇ주쌤
def get_superlet_multi_channel(x, fs, order_min, order_max, c_1, foi, clip, **kwargs):
    """
    Compute the superlet transform of the input signal.
    
    Parameters
    ----------
    x : ndarray
    fs : int
        Sampling frequency of the input signal.
    order_min : int
        Minimum order of the superlet transform.
    order_max : int
        Maximum order of the superlet transform.
    c_1 : int
        Parameter for the superlet transform.
    foi : ndarray
        Frequencies of interest in Hz.
    clip : int
        Number of time bins to clip from the beginning and end of the time axis (to handle boundary effects (artifects and so on))
    **kwargs : dict
        Additional keyword arguments to pass to the superlet function.
    
    Returns
    -------
    t : ndarray
        Array of segment times.
    foi : ndarray
        Frequencies of interest in Hz.
    spec : ndarray
        Superlet transform of x.
    """
    def scale_from_period(period):
        return period / (2 * np.pi)
    
    # Define the scales from the frequencies of interest
    scales = scale_from_period(1 / foi)
    
    # Compute the superlet transform
    specs = []

    # superlet 내부에서 window로 나누지 않으니 window로 나눠서 데이터 처리하도록
    nperseg = nperseg
    noverlap = noverlap
    window = np.hanning(nperseg)

    step = nperseg - noverlap
    shape = (x.shape[2] - noverlap) // step, nperseg
    strides = x.strides[2] * step, x.strides[2]

    segments = np.lib.stride_tricks.as_strided(x, shape=(x.shape[0], x.shape[1], (x.shape[2] - noverlap) // step, nperseg), strides=(x.strides[0], x.strides[1], x.strides[2] * step, x.strides[2]))
    windowed_segments = segments * window[np.newaxis, np.newaxis, :]

    for channel in range(x.shape[1]):
        data = windowed_segments[:, channel, :, :] # e.g., (78, 44, 256)
        chan_spec = []
        for i in range(windowed_segments.shape[0]):
            transposed_data = windowed_segments[i, 0, :, :].T

            spec = superlet(
                transposed_data,
                samplerate=fs,
                scales=scales,
                order_max=order_max,
                order_min=order_min,
                c_1=c_1,
                adaptive=True,
                **kwargs
            )
            chan_spec.append(spec)

        spec = np.abs(spec)
        spec = stats.zscore(spec[:,:,clip:-clip], axis=-1)
        spec = np.nan_to_num(spec)
        spec = torch.FloatTensor(spec)
        specs.append(spec)
    combined_spec = np.stack(specs, axis=2)
    final_spec = np.transpose(combined_spec, (1, 2, 0, 3))

    return final_spec

#! added by 스ㅇ주쌤
#below get_badnpower is shit... not correct code and such! 
# def get_bandpower(x, batch_dim=False):
#     """
#     Compute the bandpower of the input signal.
    
#     Parameters
#     ----------
#     x : ndarray
#         Input signal of shape [n, channel, freq_bin, time_len].
    
#     Returns
#     -------
#     bandpower : ndarray
#         Bandpower of the input signal.
#     """
#     assert x.ndim >= 3, "x must have shape [n, channel, freq_bin, time_len] (if time-wise average, time_len=1)"
    
#     # Define the frequency bands
#     bands = {
#         "delta": (0.5, 4),
#         "theta": (4, 8),
#         "alpha": (8, 13),
#         "beta": (13, 30),
#         "low_gamma": (30, 60),
#         "high_gamma": (60, 150) # x.shape[2] is the number of frequency bins
#     }

#     # Initialize array to hold bandpower results for each channel
#     bandpower_array = np.zeros((x.shape[0], x.shape[1], len(bands), x.shape[3]))
    
#     for sample in range(x.shape[0]):
#         for channel in range(x.shape[1]):
#             for i, band in enumerate(bands.values()):
#                 # Frequency band condition
#                 start_idx = int(band[0])
#                 end_idx = int(band[1])
#                 sliced_x = x[sample, channel, start_idx:end_idx, :]
#                 bandpower = np.mean(sliced_x, axis=0, keepdims=True)
#                 bandpower_array[sample, channel, i, :] = bandpower
    
#     return bandpower_array