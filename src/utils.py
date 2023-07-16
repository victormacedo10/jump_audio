import numpy as np
from scipy import signal
from scipy.io import wavfile

def read_audio_file(file_path, target_rate=44100):
    audio_fs, audio = wavfile.read(file_path)
    tmp = list(audio.shape)
    if len(tmp) > 1:
        audio = audio[:, 0]
    audio = normalize_wav(audio)
    if audio_fs != target_rate:
        number_of_samples = round(len(audio) * float(target_rate) / audio_fs)
        audio = signal.sps.resample(audio, number_of_samples)
        audio_fs = target_rate
    return audio, audio_fs

def normalize_wav(audio):
    if type(audio[0]) is np.int32:
        return (audio / 2147483647).astype('float32')
    elif type(audio[0]) is np.int16:
        return (audio / 32767).astype('float32')
    elif type(audio[0]) is np.uint8:
        return ((audio.astype('float32') / 127.5) - 1)
    else:
        return audio

def filter_signal(sig, fs, band_type, filter_type, cutoff_freq_Hz, N):
    num, den = signal.iirfilter(N, cutoff_freq_Hz, btype=band_type, analog=False, ftype=filter_type, fs=fs, output='ba')
    filtered_sig = signal.filtfilt(num, den, sig)
    return filtered_sig

def signal_energy(sig, fs, window_s=0.003):
    energy = np.zeros(sig.shape)
    window_size = int(window_s * fs)
    for i in range(len(sig)):
        if i >= window_size:
            energy[i] = np.mean(np.power(sig[i-window_size+1:i+1], 2))
        else:
            energy[i] = np.mean(np.power(sig[:i+1], 2))
    return energy

def custom_peak_detection(signal, upper_threshold, lower_threshold):
    # Initialize the list of detected peaks
    peaks = []
    # Initialize state variables
    searching_upper = True
    upper_crossing = None
    # Iterate through the signal
    for i in range(len(signal) - 1):
        if searching_upper:
            # Check for upper threshold crossing
            if signal[i] <= upper_threshold < signal[i + 1]:
                upper_crossing = i + 1
                searching_upper = False
        else:
            # Check for lower threshold crossing
            if signal[i] >= lower_threshold > signal[i + 1]:
                # Find the index of the maximum value in the region between the upper and lower threshold crossings
                peak = np.argmax(signal[upper_crossing:i+1]) + upper_crossing
                # Add the peak to the list of detected peaks
                peaks.append(peak)
                # Reset state variables
                searching_upper = True
    return np.array(peaks)