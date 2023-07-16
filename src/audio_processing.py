import numpy as np
import librosa
from scipy import interpolate
import noisereduce as nr
from .utils import custom_peak_detection, filter_signal, signal_energy
from .visualizations import plot_flight_time_estimation
from . import parameters as params

def noise_reduction(audio, fs, noise_sample_s):
    noise_samples = int(round(noise_sample_s * fs))
    audio = nr.reduce_noise(y=audio, sr=fs, y_noise=audio[:noise_samples+1])
    return audio

def calculate_spectral_columns(l_spec, l_spec_time, time, fs):
    # Calculate spectral columns signal
    spec_columns = np.sum(l_spec, axis=0)
    # Interpolate adjusted spectral flux signal to 44100 Hz
    tck = interpolate.splrep(l_spec_time, spec_columns, s=0)
    spec_columns_interp = interpolate.splev(time, tck, der=0)
    spec_columns_interp = (spec_columns_interp - np.min(spec_columns_interp)) / (np.max(spec_columns_interp) - np.min(spec_columns_interp))
    return spec_columns_interp

def linear_spectogram(audio, time):
    # Calculate linear spectogram
    l_spec = np.abs(librosa.stft(audio))
    # Adjusted time for the spectral flux signal
    l_spec_time = np.linspace(0, time[-1], l_spec.shape[1])
    return l_spec, l_spec_time

def find_spec_tail(l_spec, spec_peaks, l_spec_time, time, fs, window_s=0.3, window_hz=1000):
    time_bin_s = time[-1] / l_spec.shape[1]
    w_idxs = round(window_s / time_bin_s)
    freq_bin_hz = (fs / 2) / l_spec.shape[0]
    f_idxs = round(window_hz / freq_bin_hz)
    t_energy_vec = []
    for spec_peak in spec_peaks:
        spec_peak_ims = np.argmin(np.abs(l_spec_time - time[spec_peak]))
        if spec_peak_ims+w_idxs >= l_spec.shape[1]:
            t_energy = np.sum(np.power(l_spec[:f_idxs, spec_peak_ims:], 2))
        else:
            t_energy = np.sum(np.power(l_spec[:f_idxs, spec_peak_ims:spec_peak_ims+w_idxs], 2))
        t_energy_vec.append(t_energy)
    return t_energy_vec

def fine_tune_landing(lnd_peak, audio, time, fs):
    flight_sample = lnd_peak - round(params.return_time_landing_peak*fs)

    lp_sig = filter_signal(audio[flight_sample:lnd_peak], fs, band_type='lowpass', filter_type='butter', 
                        cutoff_freq_Hz=params.low_pass_cutoff_freq, N=params.low_pass_order)
    lp_energy = signal_energy(lp_sig, fs)

    idx = 0
    amp = np.max(lp_energy)
    threshold = amp * 0.1
    while(lp_energy[idx] < threshold):
        idx += 1
    # fine tuning
    lower_limit = idx - int(params.return_time_fine_tuning*fs)
    if lower_limit < 0:
        lower_limit = 0
    sig_diff = np.diff(lp_energy[lower_limit:idx+2])*fs
    diff_thres = np.max(sig_diff) * params.diff_pct_thres
    i = 0
    while(sig_diff[i] < diff_thres):
        i += 1
        if i >= len(sig_diff):
            break
    idx = i + lower_limit
    
    landing_idx = flight_sample + idx - 1
    landing_t = time[landing_idx]
    return landing_t, landing_idx

def takeoff_feature_high(ims, init_idx_ims, end_idx_ims, fs, factor=2, spec_thres=-70):
    freq_bin_hz = (fs / 2) / ims.shape[0]
    freq_scale = np.linspace(0, fs/2, ims.shape[0])
    to_search_spec = np.where(ims[:, init_idx_ims:end_idx_ims] >= spec_thres, 1, 0)
    col_heights = []
    for col in range(to_search_spec.shape[1]):
        col_height_inv = np.argmax(to_search_spec[:, col])
        if to_search_spec[col_height_inv, col] > 0:
            col_heights.append((to_search_spec.shape[0] - 1) - col_height_inv)
        else:
            col_heights.append(0)
    col_max_idx = np.max(col_heights)
    col_max_hz = freq_scale[col_max_idx]
    window_hz = col_max_hz / factor
    f_low = round(window_hz / freq_bin_hz)
    window_hz = col_max_hz
    f_high = round(window_hz / freq_bin_hz)
    to_feat_ims_high = np.sum(to_search_spec[f_low:f_high, :], axis=0) / ims.shape[0]
    if np.max(to_feat_ims_high) == np.min(to_feat_ims_high) or np.sum(to_feat_ims_high) == np.nan:
        return np.ones(len(to_feat_ims_high))
    to_feat_ims_high = (to_feat_ims_high - np.min(to_feat_ims_high)) / (np.max(to_feat_ims_high) - np.min(to_feat_ims_high))
    return to_feat_ims_high

def takeoff_feature_low(ims, init_idx_ims, end_idx_ims, fs, window_hz=1000):
    freq_bin_hz = (fs / 2) / ims.shape[0]
    f_idxs = round(window_hz / freq_bin_hz)
    to_feat_ims_low = np.sum(ims[:f_idxs, init_idx_ims:end_idx_ims], axis=0) / ims.shape[0]
    if np.max(to_feat_ims_low) == np.min(to_feat_ims_low) or np.sum(to_feat_ims_low) == np.nan:
        return np.ones(len(to_feat_ims_low))
    to_feat_ims_low = (to_feat_ims_low - np.min(to_feat_ims_low)) / (np.max(to_feat_ims_low) - np.min(to_feat_ims_low))
    return to_feat_ims_low

def takeoff_feature(spec_columns, l_spec, time, fs, landing_idx):
    ims = librosa.amplitude_to_db(l_spec, ref=np.max)
    ts = (time[-1] - time[0]) / l_spec.shape[1]
    time_center = np.linspace(time[0] + ts/2, time[-1] - ts/2, l_spec.shape[1])
    lnd_idx_ims = np.argmin(np.abs(time_center - time[landing_idx]))

    init_idx_ims = lnd_idx_ims - round(params.upper_lim_s/ts)
    end_idx_ims = lnd_idx_ims - round(params.lower_lim_s/ts)
    to_search_time = time_center[init_idx_ims:end_idx_ims]

    to_feat_ims_low = takeoff_feature_low(ims, init_idx_ims, end_idx_ims, fs)
    to_feat_ims_high = takeoff_feature_high(ims, init_idx_ims, end_idx_ims, fs)
    to_feat_ims = to_feat_ims_low * to_feat_ims_high
    init_idx = landing_idx - round(params.upper_lim_s*fs)
    end_idx = landing_idx - round(params.lower_lim_s*fs)
    tck = interpolate.splrep(to_search_time, to_feat_ims, s=0, k=1)
    to_feat = interpolate.splev(time[init_idx:end_idx], tck, der=0)
    to_feat = (to_feat - np.min(to_feat)) / (np.max(to_feat) - np.min(to_feat))
    merge_feat = to_feat * spec_columns[init_idx:end_idx]
    merge_feat = (merge_feat - np.min(merge_feat)) / (np.max(merge_feat) - np.min(merge_feat))
    return merge_feat, init_idx, end_idx

def takeoff_feature_tmp(spec_columns, l_spec, time, fs, landing_idx):
    ims = librosa.amplitude_to_db(l_spec, ref=np.max)
    ts = (time[-1] - time[0]) / l_spec.shape[1]
    time_center = np.linspace(time[0] + ts/2, time[-1] - ts/2, l_spec.shape[1])
    lnd_idx_ims = np.argmin(np.abs(time_center - time[landing_idx]))

    init_idx_ims = lnd_idx_ims - round(params.upper_lim_s/ts)
    end_idx_ims = lnd_idx_ims - round(params.lower_lim_s/ts)
    to_search_time = time_center[init_idx_ims:end_idx_ims]

    to_feat_ims_low = takeoff_feature_low(ims, init_idx_ims, end_idx_ims, fs, window_hz=params.low_freq_window)
    to_feat_ims_high = takeoff_feature_high(ims, init_idx_ims, end_idx_ims, fs, spec_thres=params.spec_thres)
    to_feat_ims = to_feat_ims_low * to_feat_ims_high
    init_idx = landing_idx - round(params.upper_lim_s*fs)
    end_idx = landing_idx - round(params.lower_lim_s*fs)
    tck = interpolate.splrep(to_search_time, to_feat_ims, s=0, k=1)
    to_feat = interpolate.splev(time[init_idx:end_idx], tck, der=0)
    to_feat = (to_feat - np.min(to_feat)) / (np.max(to_feat) - np.min(to_feat))
    merge_feat = to_feat * spec_columns[init_idx:end_idx]
    merge_feat = (merge_feat - np.min(merge_feat)) / (np.max(merge_feat) - np.min(merge_feat))
    return merge_feat, to_feat_ims_low, to_feat_ims_high

def search_takeoff(l_spec, spec_columns, landing_idx, time, fs):
    to_feat, init_idx, _ = takeoff_feature(spec_columns, l_spec, time, fs, landing_idx)
    idx = np.argwhere(to_feat > params.takeoff_threshold)[-1][0]
    time_to_return = params.return_time_takeoff
    return_idx = idx - round(time_to_return*fs)
    if return_idx < 0:
        return_idx = 0
    takeoff_idx = np.argmax(to_feat[return_idx:idx]) + return_idx + init_idx
    takeoff_t = time[takeoff_idx]
    return takeoff_t, takeoff_idx

def estimate_flight_time(audio, fs):
    time = np.linspace(0, (len(audio)-1)*(1/fs), len(audio))
    # Apply noise reduction filter based on 500 ms sample
    audio = noise_reduction(audio, fs, noise_sample_s=params.noise_sample_s)
    # Calculate linear spectogram
    l_spec, l_spec_time = linear_spectogram(audio, time)
    # Calculate spectral flux signal with hop length as parameter
    spec_columns = calculate_spectral_columns(l_spec, l_spec_time, time, fs)
    # Set spectral flux signal mean a threshold for peak detection
    spec_thres = np.mean(spec_columns)
    # Apply custom peak detection algorithm on spectral flux signal
    spec_peaks = custom_peak_detection(spec_columns, spec_thres + params.threshold_tol, spec_thres)
    # Check if at least two peaks were detected
    if len(spec_peaks) > 1:
        # Find peak tail as landing
        tails_energy = find_spec_tail(l_spec, spec_peaks, l_spec_time, time, 
                                    fs, window_s=params.tail_window_s, window_hz=params.tail_window_hz)
        # Get detected tail idx
        detected_tail_idx = np.argmax(tails_energy)
        # If tail max is the first peak, set it to the next highest tail peak
        if detected_tail_idx == 0:
            detected_tail_idx = np.argmax(tails_energy[1:]) + 1
        # Set landing peak as the peak with max tail energy
        lnd_peak = spec_peaks[detected_tail_idx]
        landing_t, landing_idx = fine_tune_landing(lnd_peak, audio, time, fs)
        takeoff_t, _ = search_takeoff(l_spec, spec_columns, landing_idx, time, fs)
    else: # If not enought peaks were detected, send warning message
        print('Not enough peaks detecting in signal, try again in a different environment or use an adhesive tape')
        return None

    flight_time = landing_t - takeoff_t

    return flight_time, takeoff_t, landing_t

def jump_height_from_audio(audio, fs):
    result = estimate_flight_time(audio, fs)
    if result:
        flight_time, takeoff_t, landing_t = result
        time = np.linspace(0, (len(audio)-1)*(1/fs), len(audio))
        jump_height = np.round((params.g * np.power(flight_time, 2))/ 8 * 100, 2)
        plot_flight_time_estimation(time, audio, jump_height, takeoff_t, landing_t)
        return jump_height
    else:
        return -1