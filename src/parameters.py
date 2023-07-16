# gravitational acceleration constant m/s^2
g = 9.7838
# maximum considered flight time (s) 
upper_lim_s = 0.8
# minimum considered flight time (s)
lower_lim_s = 0.2
# difference percentage threshold for landing fine tuning
diff_pct_thres = 0.8
# time to return from landing peak to search for landing instant
return_time_landing_peak = 0.1
# time to return to precede threshold detection to fine tune landing instant
return_time_fine_tuning = 0.03
# time to return to precede threshold detection takeoff
return_time_takeoff = 0.05
# low pass filter cutoff frequency
low_pass_cutoff_freq = 3000
# low pass filter order
low_pass_order = 2
# durantion in seconds of the initial noise sample to analyze for noise reduction
noise_sample_s = 0.5
# tail energy time window in seconds for landing peak classification
tail_window_s = 0.3
# tail energy frequency window in hertz for landing peak classification
tail_window_hz = 500
# threshold tolerance for upper limit peak detection on spectral columns
threshold_tol = 0.1
# threshold for detecting takeoff event on feature
takeoff_threshold = 0.1
# threshold for counting cells in spectogram
spec_thres = -70
# window in hertz for computing low frequency feature
low_freq_window = 1000