import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Generate continuous signal, using sum of low frequency sinusoids
#Need an input for target power, option 1 or 2, and alpha and number of components
def generate_signal(n_components,
                    target_power,
                    alpha, option,
                    T = 5.0,
                    fs_cont = 10000,
                    f_max = 20):
    
    t = np.linspace(0, T, int(T*fs_cont), endpoint=False)

    frequencies = np.random.uniform(1, f_max, n_components)
    phases = np.random.uniform(0, 2*np.pi, n_components)

    if option == 1:
    # Flat/random amplitudes
        amplitudes= np.random.uniform(0.5, 1.0, n_components)

    elif option ==2:
        # Spectral decay amplitudes
        amplitudes = 1 / (frequencies ** alpha)

        # Slight randomness to avoid perfectly deterministic structure
        amplitudes = 1 / (frequencies ** alpha) * np.random.uniform(0.9, 1.1, n_components)

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

    return t, signal, current_power


# 2. Reconstruction method (Non-Uniform Discrete Fourier Transform (Type 1))
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

    if return_phase:
        return np.abs(X), np.angle(X)
    else:
        return np.abs(X)


# 3. Defining noise
def wavelength_in_samples(fs_cont, frequency):
    wavelength = fs_cont / frequency
    
    return (wavelength)
def low_freq_sin_noise(t, amplitude,
                                   frequency, T = 5.0,
                                   fs_cont = 10000):

    low_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_1 = np.mean(low_freq_sin**2)

    # Call function to work out wavelength
    wavelength_1 = wavelength_in_samples(fs_cont, frequency)

    return low_freq_sin, current_power_1, wavelength_1

def high_freq_sin_noise(t, amplitude,
                                   frequency, T = 5.0,
                                   fs_cont = 10000):
    
    high_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_2 = np.mean(high_freq_sin**2)
    
    # Call function to work out wavelength
    wavelength_2 = wavelength_in_samples(fs_cont, frequency)

    return high_freq_sin, current_power_2, wavelength_2

def multi_tone_sin_noise(t, amplitude_min,
                                   amplitude_max,
                                   frequency_min,
                                   frequency_max, num_sin,
                                   T = 5.0, fs_cont = 10000):

    amplitudes = np.random.uniform(amplitude_min, amplitude_max, num_sin)
    frequencies = np.random.uniform(frequency_min, frequency_max, num_sin)

    # Build combined sine wave signal
    multi_tone_sin = np.zeros_like(t)
    for A, f in zip(amplitudes, frequencies):
        multi_tone_sin += A * np.sin(2 * np.pi * f * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_3 = np.mean(multi_tone_sin**2)

    return multi_tone_sin, current_power_3

                           
# Acts as the control
def white_gaussian_noise(t, amplitude):
    white_noise = amplitude * np.random.randn(len(t))

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_4 = np.mean(white_noise**2)

    return white_noise, current_power_4


# 4. Sampling
# Prime Modulus Sampling

# Co-Prime Sampling

# Random Prime Sampling

# Multiple Prime Step Sampling

# Uniform Sampling

# Random Uniform Sampling

# Jittered Uniform Sampling


# 5. Matplotlib of Continuous signal and Magnitude spectrum
def graph_Csignal_Mspec (t, signal, frequency_grid, X):
        # ---- Plot Time-Domain Signal ----
    plt.figure(figsize=(20, 4))
    plt.plot(t, signal)
    plt.title("Continuous Time-Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Plot Magnitude Spectrum ----
    plt.figure(figsize=(20, 4))
    plt.plot(frequency_grid, X)
    plt.title("NUDFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 6 Matplotlib of Noisy signal and Noisy Magnitude spectrum overlaid with original
def graph_noisy_vsignal (t, signal, noisy_signal, frequency_grid, X, Y):
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
    plt.plot(frequency_grid, X, "r", label = "Original Magnitude Spectrum")
    plt.plot(frequency_grid, Y, "b", label = "Noisy Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()



#Main section
if __name__ == "__main__":

    # ---- Parameters ----
    n_components = int(input("Please input the number of components for the sinusoid (recommended 8-12 for moderate complexity) (A realistic signal is 20-50):  "))
    target_power = float(input("Please input target power, recommended 1.0:  "))
    try:
        option = int(input("Please input 1 = flat amplitudes, 2 = spectral decay:  "))
    except:
        raise ValueError("Must choose options 1 or 2")
    if option == 2:
        alpha = float(input("Please input alpha value for spectral decay:  "))

    else:
        alpha = 1
    
    # Start of timer
    start = time.time()
    # ---- Generate Signal ----
    t, signal, current_power = generate_signal(
        n_components,
        target_power,
        alpha, option)

    # Frequency grid for NUDFT
    frequency_grid = np.linspace(0, 50, 3000)

    # Compute Spectrum
    X = NUDFT_reconstruction(signal, t, frequency_grid)
    # End of timer
    end = time.time()

    # Call Graph for continuous signal and Magnitude spectrum
    graph_Csignal_Mspec (t, signal, frequency_grid, X)

    # Prints out the noise level
    current_power = round(current_power, 5)
    print("The power level is currently",current_power)

    # Works out length of time and prints it, helps to work out efficiency
    length = end - start
    length = round(length,3)
    print("The time taken was", length,"seconds, to generate the signal and compute NUDFT.")

    # Option for noise injection
    noisy_signal = signal.copy()
    noise_opt = 0

    while noise_opt != 4:
        noise_opt = int(input("Input noise you want, (4 stops adding noise) (0 = L_f_s noise) (1 = H_f_s noise) (2 = M_t_s noise) (3 = w_g noise):  ") )
        frequency = 0.0
        amplitude = 0.0
        amplitude_min = 0.0
        amplitude_max = 0.0
        frequency_min = 0.0
        frequency_max = 0.0
        num_sin = 0
        
        try:
            if noise_opt == 0:
                amplitude = float(input("Please input an amplitude, recommended 0.2 - 0.4:  "))
                frequency = float(input("Please input a frequency, recommended 0.5 - 1 Hz:  "))

                noise, _ , wavelength_1 = low_freq_sin_noise(t, amplitude, frequency)
                noisy_signal += noise

                print("Wavelength is", round(wavelength_1, 3), "samples / wavelength")

            elif noise_opt == 1:
                amplitude = float(input("Please input an amplitude, recommended 0.2 - 0.4:  "))
                frequency = float(input("Please input a frequency, recommended 25 - 40 Hz:  "))
                
                noise, _ , wavelength_2 = high_freq_sin_noise(t, amplitude, frequency)
                noisy_signal += noise

                print("Wavelength is", round(wavelength_2, 3), "samples / wavelength")

            elif noise_opt == 2:
                amplitude_min = float(input("Please input a minimum amplitude, recommended 0.2:  "))
                amplitude_max = float(input("Please input a maximum amplitude, recommended 0.4:  "))
                frequency_min = float(input("Please input a minimum frequency, recommended 25 Hz:  "))
                frequency_max = float(input("Please input a maximum frequency, recommended 40Hz:  "))
                num_sin = int(input("Please input the number of sin waves added as noise:  "))

                noise, _ = multi_tone_sin_noise(t, amplitude_min,
                                                                amplitude_max,
                                                                frequency_min,
                                                                frequency_max, num_sin)
                noisy_signal += noise

                print("Wavelength can't be worked outer here (sum of multiple sine waves)")

            elif noise_opt == 3:
                # Recommended to be amplitude 0.2
                amplitude = float(input("Please input a float, recommended 0.2:  "))
                
                noise, _ = white_gaussian_noise(t, amplitude)
                noisy_signal += noise

        except:
            raise ValueError("Must choose options 0-4")

    # Frequency grid for NUDFT
    frequency_grid = np.linspace(0, 50, 3000)

    # Compute Spectrum (called Y so can use X later on)
    Y = NUDFT_reconstruction(noisy_signal, t, frequency_grid)

    # Call Graph for noisy signal and Magnitude spectrum
    graph_noisy_vsignal (t, signal, noisy_signal, frequency_grid, X, Y)

    # Working out and printing the difference in power
    relative_noise_power = np.mean((noisy_signal - signal) **2) / np.mean(signal ** 2)
    print("The relative noise power is", round(relative_noise_power,4))

    # Working out the noise level from the power ratio of noisy signal to signal
    noise_db = 10 *np.log10(relative_noise_power)
    print("Relative noise level:", round(noise_db,3), "dB")
