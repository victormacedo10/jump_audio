import matplotlib.pyplot as plt

def plot_flight_time_estimation(time, audio, jump_height, takeoff_t, landing_t):
    flight_time = int(round(1000 * (landing_t - takeoff_t)))
    fig, audio_axis = plt.subplots(figsize=(9,5))
    audio_axis.plot(time, audio, color='#929A97')
    audio_axis.axvspan(takeoff_t, landing_t, color="#65BF9E", alpha=0.6, label='flight')
    audio_axis.set_xlabel('Time (s)')
    plt.title(f'Estimated jump height: {jump_height:.2f} cm - Flight time: {flight_time} ms')
    plt.grid(True)
    plt.legend()
    plt.show()