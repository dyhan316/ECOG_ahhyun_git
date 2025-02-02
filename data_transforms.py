import numpy as np
from scipy import signal, stats


def normalize_memory_task(data, demean_only = False, option = "full_trial"): 
    #!assumes shape of (3,120, 10000) (sub, subtype, time) or (3, 120, 20, 10000) (sub, subtype, trial, time)
    #! also assumes sampling rate of 2000Hz, and 2seconds must be removed from the front if fixation
    if option == "full_trial" :  #noramlize using the whole trial
        if demean_only : 
            return data - np.mean(data, axis = -1, keepdims = True)
        else : 
            return (data - np.mean(data, axis = -1, keepdims = True)) / np.std(data, axis = -1, keepdims = True)
    elif option == "fixation" : #normalize using the first 2 seconds (fixation timepoints)
        mean_arr = np.mean(data[:,:,:2*2000], axis = -1, keepdims = True) if len(data.shape) == 3 else np.mean(data[:,:2*2000], axis = -1, keepdims = True)
        std_arr = np.std(data[:,:,:2*2000], axis = -1, keepdims = True) if len(data.shape) == 3 else np.std(data[:,:2*2000], axis = -1, keepdims = True)
        if demean_only : 
            return data - mean_arr
        else :
            return (data - mean_arr) / std_arr


###getting STFT and such 
def get_stft_multi_channel(x, fs, clip_fs=-1, normalizing=None, boundary_clip=5, batch_dim=False, **kwargs):
    raise NotImplementedError("This function is not implemented yet. Use get_stft instead.")
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

def parallel_get_stft(x, fs, clip_fs=-1, normalizing=None, **kwargs):
    r""" 
    applies get_stft to multiple channels/batches in parallel
    assumes x has shape [batch, time_len] or [batch, channel, time_len]
    the stft should be applied to each time_len 
    """
    Zxx_list = []
    
    if x.ndim == 2:  # shape [batch, time_len]
        for i in range(x.shape[0]):
            f, t, Zxx = get_stft(x[i], fs, clip_fs=clip_fs, normalizing=normalizing, **kwargs)
            Zxx_list.append(Zxx)
    elif x.ndim == 3:  # shape [batch, channel, time_len]
        for i in range(x.shape[0]):
            Zxx_channel_list = []
            for j in range(x.shape[1]):
                f, t, Zxx = get_stft(x[i, j], fs, clip_fs=clip_fs, normalizing=normalizing, **kwargs)
                Zxx_channel_list.append(Zxx)
            Zxx_list.append(np.stack(Zxx_channel_list, axis=0))
    else:
        raise ValueError("Input x must have 2 or 3 dimensions")
    
    return f, t, np.stack(Zxx_list, axis=0)
        


#! MUST KNOW THE OTHER STUFF LIKE THE NPERSEG AND SUCH TO USE! (nperseg=nperseg, noverlap=nperseg-50)
#*below version : not used since single channel version..
def get_stft(x, fs, clip_fs = None, normalizing=None, **kwargs):
    """
    #write documentation 
    fs : int
        Sampling frequency of the input signal
    clip_fs : int
        Number of frequency bins to keep (i.e. frequency clipping (clips higher frequencies))
    normalizing : str
        Type of normalization to apply to the STFT. Options are "zscore", "baselined", "db"
    **kwargs : dict
        Additional keyword arguments to pass to scipy.signal    
    """
    f, t, Zxx = signal.stft(x, fs, **kwargs)
    
    if clip_fs is None:
        clip_fs = fs // 2
    assert clip_fs > 0, "clip_fs must be greater than 0"
    assert clip_fs <= fs // 2, "clip_fs must be less than or equal to fs // 2"
    clip_fs_idx = np.argmax(f >= clip_fs)
    Zxx = Zxx[:clip_fs_idx]
    f = f[:clip_fs_idx]

    Zxx = np.abs(Zxx) #magnitude of the complex number
    
    if normalizing=="zscore":
        Zxx = stats.zscore(Zxx, axis=-1)
    elif normalizing=="baselined":
        raise NotImplementedError
    elif normalizing=="db":
        Zxx = np.log2(Zxx)
    elif normalizing is None:
        pass 

    if np.isnan(Zxx).any():
        import pdb; pdb.set_trace()

    return f, t, Zxx











##for getting stft and such! 



##get 
#1. normalized (w.r.t to the whole timepoint, or the first three, or on a trial by trial basis)
#2. STFT
#3. Bandpower (alpha, beta, gamma, theta, delta)
#3. PCA (if need be)
#4. subject-wise thing too! 
#train/test split must be done! 
#cv?  => 일단 높은 성능이 필요하면! 

#during all subject vs subject-wise model comparison, keep the samples same!! 