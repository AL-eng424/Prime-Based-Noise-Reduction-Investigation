import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
from scipy.interpolate import CubicSpline # Redundant
from scipy.fft import rfft, rfftfreq

# Global Variables, to easily change parameters
# Generating a signal
global T
global fs_cont
global f_max
global min_freq
global n_components 
global target_power
global option
global alpha

T = 5.0 # Time
fs_cont = 10000 # Uniform sampling rate (before sampling is applied)
f_max = 20 # Max frequency
min_freq = 1 # Minimum frequency, lowest it can be is 1/T
n_components = 50 # The number of sinusoids used to generate a signal
target_power = 1 # The target power of the signal
option = 2 #Option for generating signal 1 = flat amplitudes, 2 = spectral decay
alpha = 1.5

# Noise injection
global frequency
global frequency_min
global frequency_max
global amplitude
global amplitude_min
global amplitude_max
global num_sin
global target_noise_power

noise_opt = 1 # (0 = L_f_s noise) (1 = H_f_s noise) (2 = M_t_s noise) (3 = w_g noise)
frequency = 40 # Frequency for L_f_s, H_f_s # MUST BE A FACTOR OF FS_CONT
amplitude = 0.2 # Amplitude for L_f_s, H_f_s, w_g

# Multi-tone sin
amplitude_min = 0.2
amplitude_max = 0.4
frequency_min = 10
frequency_max = 30
num_sin = 3 # Number of sin waves added

# Target noise power
target_noise_power = 0.01

# Sampling
global fs_sample
global min_fs
global max_fs

sample_option = 4 #(0 =  Uniform sampling) (1 =  Random sampling) (2 = Random interval sampling) (3 = P_M_A_F_CS) (4 = P_M_A_F_LI )
# Need to be above the Niquist frequency
fs_sample = 2000 # Sampling rate for Uniform and Random sampling
min_fs = 1000 # Minimum sampling rate for Random interval sampling
max_fs = 3000 # Maximum sampling rate for Random interval sampling

# PMAF specific sampling
global w
global n

w = 13
n = 25

# 1. Generate continuous signal, using sum of low frequency sinusoids
def generate_signal():
    
    t = np.linspace(0, T, int(T*fs_cont), endpoint=False)

    frequencies = np.random.uniform(min_freq , f_max, n_components)
    phases = np.random.uniform(0, 2*np.pi, n_components)

    if option == 1:
    # Flat/random amplitudes
        amplitudes = np.random.uniform(0.5, 1.0, n_components)

    elif option ==2:
        # Slight randomness to avoid perfectly deterministic structure
        decay_factor = frequencies ** alpha
        amplitudes = np.random.uniform(0.9, 1.1, n_components) / decay_factor

    else:
        raise ValueError("option must be 1 (flat) or 2 (spectral decay)")

    # Build signal
    signal = np.zeros_like(t)
    for A, f, phi in zip(amplitudes, frequencies, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)

    # Normalise signal power
    current_power = np.mean(signal**2)
    signal *= np.sqrt(target_power / current_power)
    current_power = np.mean(signal**2)

    return t, signal, current_power, frequencies


# 2. Reconstruction method (Non-Uniform Discrete Fourier Transform (Type 3), needs to be a type 1)
def NUDFT_reconstruction(signal, t, frequencies, return_phase=False):
    signal = np.asarray(signal)
    t = np.asarray(t)
    frequencies = np.asarray(frequencies)
    
    dt = t[1] - t[0]  # Sampling interval
    T = len(t) * dt

    tc = t - t.mean()

    X = np.zeros(len(frequencies), dtype=complex)
    
    # Compute NUDFT frequency by frequency (memory-efficient)
    for k, f in enumerate(frequencies):
        exponential = np.exp(-2j * np.pi * f * tc)
        X[k] = np.dot(signal, exponential) * dt / T

    magnitude = np.abs(X)

    if len(magnitude) > 1:
        magnitude [1:] *= 2
        
    if return_phase:
        return magnitude , np.angle(X)
    else:
        return magnitude

def FFT_reconstruction(signal):
    len_signal = int(T * fs_cont)

    X = rfft(signal)

    freq = rfftfreq(len_signal, d = 1/fs_cont)
    mask = freq <= 50

    magnitude = np.abs(X) / len_signal
    if len(magnitude) > 1:
        magnitude[1:-1] *= 2


    return magnitude[mask], freq[mask], X[mask]
    

# 3. Defining noise
def low_freq_sin_noise(t):

    low_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_1 = np.mean(low_freq_sin**2)

    # Normalises_noise
    low_freq_sin *= np.sqrt(target_noise_power / current_power_1)
    current_power_1 = np.mean(low_freq_sin ** 2)

    return low_freq_sin, current_power_1

def high_freq_sin_noise(t):
    
    high_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_2 = np.mean(high_freq_sin ** 2)
    # Normalises noise
    high_freq_sin *= np.sqrt(target_noise_power / current_power_2)
    current_power_2 = np.mean(high_freq_sin ** 2)

    return high_freq_sin, current_power_2

def multi_tone_sin_noise(t):

    amplitudes = np.random.uniform(amplitude_min, amplitude_max, num_sin)
    frequencies = np.random.uniform(frequency_min, frequency_max, num_sin)

    # Build combined sine wave signal
    multi_tone_sin = np.zeros_like(t)
    for A, f in zip(amplitudes, frequencies):
        multi_tone_sin += A * np.sin(2 * np.pi * f * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_3 = np.mean(multi_tone_sin**2)
    
    # Normalises noise
    multi_tone_sin *= np.sqrt(target_noise_power / current_power_3)
    current_power_3 = np.mean(multi_tone_sin ** 2)
    
    return multi_tone_sin, current_power_3, frequencies


def white_gaussian_noise(t):
    white_noise = amplitude * np.random.randn(len(t))

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_4 = np.mean(white_noise ** 2)

    # Normalises_noise
    white_noise *= np.sqrt(target_noise_power / current_power_4)
    current_power_4 = np.mean(white_noise ** 2)

    return white_noise, current_power_4


# 4. Sampling
#  Prime Average Sampling
#def Welch_PMAF(noisy_signal):
    


# This will bias upward in Coherence analysis
def P_M_A_F_LI(noisy_signal):
    repeated_signal = np.tile(noisy_signal, n)

    t_full = np.linspace(0, n * T, len(repeated_signal), endpoint = False)
    
    samples = repeated_signal[::w]
    t_samples = t_full[::w]
    interp = np.interp(t_full, t_samples, samples)

    chunks = interp.reshape(n, int(T * fs_cont))
    reconstructed = np.mean(chunks, axis = 0)

    return reconstructed


# Uniform Sampling
def Uniform_Sampling(noisy_signal, t):
    if fs_sample > fs_cont:
        raise ValueError("Sampling frequency cannot exceed continuous frequency")

    step = round(fs_cont / fs_sample)

    indicies = np.arange(0,  len(t), step)
    
    sampled_signal = noisy_signal [indicies]
    sampled_t = t[indicies]

    return sampled_signal, sampled_t, indicies


# Random Sampling
def Random_Sampling(noisy_signal, t):
    if fs_sample > fs_cont:
        raise ValueError("Sampling frequency cannot exceed continuous frequency")

    n_samples = int(fs_sample * T)

    indicies = np.sort(np.random.choice(len(t), n_samples, replace = False))

    sampled_signal = noisy_signal[indicies]
    sampled_t = t[indicies] # In case we want to use this later to see where it sampled

    return sampled_signal, sampled_t, indicies
    

# Random interval sampling
def Random_interval_sampling(noisy_signal, t):
    if max_fs > fs_cont:
        raise ValueError("Sampling frequency cannot exceed continuous frequency")

    if min_fs <= 0 or max_fs <= 0:
        raise ValueError("Sampling frequencies must be positive")

    if min_fs > max_fs:
        raise ValueError("min_fs must be smaller than max_fs")

    max_step = int(np.floor(fs_cont / max_fs))
    min_step = int(np.ceil(fs_cont / min_fs))

    if max_step > min_step:
        raise ValueError("Invalid step bounds")
    
    indicies = []
    i = 0

    while i < len(t):
        indicies.append(i)
    
        fs_random = np.random.uniform(min_fs, max_fs)
        step = int(fs_cont / fs_random)
    
        i += step

    sampled_signal = noisy_signal[indicies]
    indicies = np.array(indicies)
    sampled_t = t[indicies]

    average_fs = len(indicies) / T

    return sampled_signal, sampled_t, indicies, average_fs
    
# 5. Analysis
def Welch_Coherence(x, y, nperseg=1024, noverlap=512):
    
    x = np.asarray(x)
    y = np.asarray(y)

    step = nperseg - noverlap
    window = np.hanning(nperseg)

    n_segments = (len(x) - noverlap) // step

    freqs = rfftfreq(nperseg, d=1/fs_cont)

    Pxx = np.zeros(len(freqs))
    Pyy = np.zeros(len(freqs))
    Pxy = np.zeros(len(freqs), dtype=complex)

    count = 0
    
    for i in range(n_segments):

        start = i * step
        end = start + nperseg

        if end > len(x):
            break

        x_seg = x[start:end] * window
        y_seg = y[start:end] * window

        X = rfft(x_seg)
        Y = rfft(y_seg)

        Pxx += np.abs(X)**2
        Pyy += np.abs(Y)**2
        Pxy += X * np.conj(Y)

        count += 1

    if count == 0:
        return freqs, np.zeros_like(freqs)

    Pxx /= count
    Pyy /= count
    Pxy /= count

    denom = Pxx * Pyy
    Cxy = np.zeros_like(denom)
    valid = denom > 0
    Cxy[valid] = (np.abs(Pxy[valid])**2) / denom[valid]

    mask = freqs <= 50
    freqs = freqs[mask]
    Cxy = Cxy[mask]

    return freqs, np.clip(Cxy, 0, 1)

# 6. Matplotlib of Continuous signal and Magnitude spectrum
def graph_Csignal_Mspec (t, signal, X_freq, X_mag):
        # Plot Time-Domain Signal
    plt.figure(figsize=(20, 4))
    plt.plot(t, signal)
    plt.title("Continuous Time-Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Magnitude Spectrum
    plt.figure(figsize=(20, 4))
    plt.plot(X_freq, X_mag)
    plt.title("NUDFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0, 51, 1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 7. Matplotlib of Noisy signal and Noisy Magnitude spectrum overlaid with original
def graph_noisy_vsignal (t, signal, noisy_signal, X_freq, Y_freq, X_mag, Y_mag):
    # Plot signal vs noisy signal
    plt.figure(figsize = (20, 4))
    plt.plot(t, signal, "r", label = "Original Signal")
    plt.plot(t, noisy_signal, "b", label = "Noisy Signal")
    plt.legend()
    plt.title("Original Signal vs Noisy Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Magnitude Spectrum
    plt.figure(figsize=(20, 4))
    plt.plot(X_freq, X_mag, "r", label = "Original Magnitude Spectrum")
    plt.plot(Y_freq, Y_mag, "b", label = "Noisy Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0, 51, 1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 8. Matplotlib graph for sampled signal and nudft
def graph_Ssignal_Mspec_FFT (sampled_signal, t,
                                    noisy_signal, signal, Y_freq, Z_freq, X_freq , Y_mag, Z_mag, X_mag):

    # Plot Sampled Signal v Noisy Signal
    plt.figure(figsize = (20, 4))
    plt.plot(t, sampled_signal, "r", label = "Sampled Signal")
    plt.plot(t, noisy_signal, "b", label = "Noisy Signal")
    plt.legend()
    plt.title("Sampled Signal vs Noisy Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Sampled Signal v Original Signal
    plt.figure(figsize = (20, 4))
    plt.plot(t, sampled_signal, "r", label = "Sampled Signal")
    plt.plot(t, signal, "b", label = "Original Signal")
    plt.legend()
    plt.title("Sampled Signal vs Original Signal")
    plt.xlabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot Sampled Magnitude Spectrum vs Noisy Magnitude Spectrum
    plt.figure(figsize = (20, 4))
    plt.plot(Z_freq, Z_mag, "r", label = "Sampled Magnitude Spectrum")
    plt.plot(Y_freq, Y_mag, "b", label = "Noisy Magnitude Spectrum")
    plt.legend()
    plt.title("FFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Sampled Magnitude Spectrum vs Original Manigutde Spectrum
    plt.figure(figsize = (20, 4))
    plt.plot(Z_freq, Z_mag, "r", label = "Sampled Magnitude Spectrum")
    plt.plot(X_freq, X_mag, "b", label = "Original Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graph_Ssignal_Mspec (sampled_t, sampled_signal, t,
                                    noisy_signal, signal, frequency_grid, Y_mag, Z, X_mag, Y_freq, X_freq):
    # Plot Sampled Signal v Noisy Signal
    plt.figure(figsize = (20, 4))
    plt.plot(sampled_t, sampled_signal, "r", label = "Sampled Signal")
    plt.plot(t, noisy_signal, "b", label = "Noisy Signal")
    plt.legend()
    plt.title("Sampled Signal vs Noisy Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Sampled Signal v Original Signal
    plt.figure(figsize = (20, 4))
    plt.plot(sampled_t, sampled_signal, "r", label = "Sampled Signal")
    plt.plot(t, signal, "b", label = "Original Signal")
    plt.legend()
    plt.title("Sampled Signal vs Original Signal")
    plt.xlabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot Sampled Magnitude Spectrum vs Noisy Magnitude Spectrum
    plt.figure(figsize = (20, 4))
    plt.plot(frequency_grid, Z, "r", label = "Sampled Magnitude Spectrum")
    plt.plot(Y_freq, Y_mag, "b", label = "Noisy Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Sampled Magnitude Spectrum vs Original Manigutde Spectrum
    plt.figure(figsize = (20, 4))
    plt.plot(frequency_grid, Z, "r", label = "Sampled Magnitude Spectrum")
    plt.plot(X_freq, X_mag, "b", label = "Original Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def Coherence_plot(Cxy, frequencies, noise_freq, Co_opt):
    plt.figure(figsize=(20,4))
    plt.plot(frequencies, Cxy, label="Coherence Cxy(f)")
    
    plt.title("Coherence between Original and Sampled Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cxy(f) (0 - 1)")
    plt.xticks(np.arange(0,51,5))
    plt.yticks(np.arange(0,1.2,0.2))
    plt.grid(True)

    # Make sure noise_freq is iterable
    if np.isscalar(noise_freq):
        noise_freq = [noise_freq]

    # Plot noise lines
    for i, f in enumerate(noise_freq):
        if i == 0:
            plt.axvline(f, color='r', linestyle='--', label=f"Injected Noise: {', '.join(map(str, noise_freq))} Hz")
        else:
            plt.axvline(f, color='r', linestyle='--')

    plt.legend()
    plt.tight_layout()
    plt.show()



#########_____________MAIN SECTION_____________#########




# Generate Signal
t, signal, current_power, frequencies = generate_signal() # Never actually use frequencies so get rid of it?

# Frequency grid for NUDFT
# Old code frequency_grid = np.linspace(0, 50, 3000)

# Compute Spectrum
#old code X = NUDFT_reconstruction(signal, t, frequency_grid)
X_mag, X_freq, X = FFT_reconstruction(signal)

# Call Graph for continuous signal and Magnitude spectrum
graph_Csignal_Mspec (t, signal, X_freq, X_mag)

# Prints out the noise level
current_power = round(current_power, 5)
print("The power level in original signal is:", current_power)

# Option for noise injection
noisy_signal = signal.copy()

if noise_opt == 0:
# Recommended: amplitude 0.2-0.4, frequency 0.5-1
    noise, _  = low_freq_sin_noise(t)
    noisy_signal += noise

    frequencies = np.append(frequencies, frequency)

elif noise_opt == 1:
# Recommended: amplitude 0.2-0.4, frequency 25-40
    noise, _ = high_freq_sin_noise(t)
    noisy_signal += noise

    frequencies = np.append(frequencies, frequency)

elif noise_opt == 2:
# Recommended: amplitude (min = 0.2, max = 0.4) frequency (abs min = 1 / T, abs max = 1/2 * fs_cont)
    noise, _, noise_frequencies = multi_tone_sin_noise(t)
    noisy_signal += noise

    frequencies = np.concatenate((frequencies, noise_frequencies))
    
elif noise_opt == 3:
    # Recommended: amplitude = 0.2

    noise, _ = white_gaussian_noise(t, amplitude, target_noise_power)
    noisy_signal += noise

# Frequency grid for NUDFT
# Old code frequency_grid = np.linspace(0, 50, 3000)

# Compute Spectrum (called Y so can use X later on)
# Old code Y = NUDFT_reconstruction(noisy_signal, t, frequency_grid)
Y_mag, Y_freq, _ = FFT_reconstruction(noisy_signal)

# Call Graph for noisy signal and Magnitude spectrum
graph_noisy_vsignal (t, signal, noisy_signal, X_freq, Y_freq, X_mag, Y_mag)

# Working out and printing the relative noise power
relative_noise_power = np.mean((noisy_signal - signal) **2) / np.mean(signal ** 2)
print("The relative noise power before sampling is:", round(relative_noise_power, 5))

# Working out the noise level from the power ratio of noisy signal to signal
noise_db = 10 *np.log10(relative_noise_power)
print("Relative noise level before sampling is:", round(noise_db, 5), "dB")

# Working out and printing the noise power and current signal power
noise_power = np.mean((signal - noisy_signal) ** 2)
print("The power in the noise is before sampling is:", round(noise_power, 5))
      
noisy_signal_power = np.mean(noisy_signal ** 2)
print("The current noisy signal power is:", round(noisy_signal_power, 5))

# Sampling
if sample_option == 0:
    sampled_signal, sampled_t, indicies = Uniform_Sampling(noisy_signal, t)

elif sample_option == 1:
    sampled_signal, sampled_t, indicies = Random_Sampling(noisy_signal, t)

elif sample_option == 2:
    sampled_signal, sampled_t, indicies, average_fs = Random_interval_sampling(noisy_signal, t)

    print("The average sampling rate is:", average_fs)

elif sample_option == 3:
    sampled_signal = Welch_PMAF(noisy_signal)

elif sample_option == 4:
    sampled_signal = P_M_A_F_LI(noisy_signal)
    
# If not PMAF
if sample_option != 3 and sample_option != 4:
    # NUDFT of sampled signal
    frequency_grid = np.linspace(0, 50, 3000)
    Z = NUDFT_reconstruction(sampled_signal, sampled_t, frequency_grid)

    # Matplotlib comparing sampled signal to original signal to noisy signal
    graph_Ssignal_Mspec (sampled_t, sampled_signal, t,
                                    noisy_signal, signal, frequency_grid, Y_mag, Z, X_mag, Y_freq, X_freq) # Need Y_freq, X_freq etc, so need function to be changed
    # Work out the sampled noise and percentage reduction
    sampled_power = np.mean(sampled_signal ** 2)
    print("The power in the sample signal is:", round(sampled_power, 5))

    sampled_noise_power = np.mean((signal[indicies] - sampled_signal) ** 2)
    print("The noise power in the sampled signal is:", round(sampled_noise_power, 5))

    percent_reduction_noise = (sampled_noise_power / noise_power) * 100
    print("The percentage of noise left is:", str(round(percent_reduction_noise, 3)) + "%")

# If PMAF / Welch style PMAF
elif sample_option == 3 or sample_option == 4:
        # FFT of sampled signal
    # Old codefrequency_grid = np.linspace(0, 50, 3000)
    # Old code Z = NUDFT_reconstruction(sampled_signal, t, frequency_grid)
    Z_mag, Z_freq, Z = FFT_reconstruction(sampled_signal)
    # Matplotlib comparing sampled signal to original signal to noisy signal
    graph_Ssignal_Mspec_FFT (sampled_signal, t,
                                    noisy_signal, signal, Y_freq, Z_freq, X_freq, Y_mag, Z_mag, X_mag)

    # Work out the sampled noise and percentage reduction
    sampled_power = np.mean(sampled_signal ** 2)
    print("The power in the sample signal is:", round(sampled_power, 5))

    sampled_noise_power = np.mean((signal - sampled_signal) ** 2)
    print("The noise power in the sampled signal is:", round(sampled_noise_power, 5))

    percent_reduction_noise = (sampled_noise_power / noise_power) * 100
    print("The percentage of noise left is:", str(round(percent_reduction_noise, 3)) + "%")

    # This is probably wrong
    # Works out noise floor
    noise_floor = noise_power/ n

    # Works out how much noise it was possible to reduce and prints a percentage of 
    possible_noise_reduction = 100 *((noise_power - sampled_noise_power) / (noise_power - noise_floor))
    print("Percentage of noise removed compared to noise that was possible to remove is:", str((round(possible_noise_reduction, 3))) + "%")

# Coherence analysis
# Need to work out scaling for NUDFT APSD coeficients (scaled by dt / T)

if sample_option == 3 or sample_option == 4:
    freqs, Cxy = Welch_Coherence(signal, sampled_signal)


    if noise_opt == 2:
            Co_opt = 1
            Coherence_plot(Cxy, freqs, noise_frequencies, Co_opt)

    else:
        Co_opt = 2
        Coherence_plot(Cxy, freqs,  frequency, Co_opt)
    
    
elif sample_option != 3 and sample_option != 4:
    print("You still need to do the NUDFT branch of Coherence")
