# Next thing to implement, list of all frequencies with their magnitude, so when you click on peak / 
# or something else it shows you the frequency and magnitude
# Remove the wavelength thing
# Write more of the sampling functions
# Write more analysis functions
# Once finished rewrite main section for monte carlo analysis


import numpy as np
import matplotlib.pyplot as plt
import time
from sympy import primerange
# Global Variables, to easily change parameters
# Do this for all inputs? (most likely better if we want to do monte carlo runs)
global T
global f_max
global fs_cont

T = 5.0
f_max = 20
fs_cont = 10000


# 1. Generate continuous signal, using sum of low frequency sinusoids
#Need an input for target power, option 1 or 2, and alpha and number of components
def generate_signal(n_components,
                    target_power,
                    alpha, option, min_freq):
    
    t = np.linspace(0, T, int(T*fs_cont), endpoint=False)

    frequencies = np.random.uniform(min_freq , f_max, n_components)
    phases = np.random.uniform(0, 2*np.pi, n_components)

    if option == 1:
    # Flat/random amplitudes
        amplitudes= np.random.uniform(0.5, 1.0, n_components)

    elif option ==2:
        # Slight randomness to avoid perfectly deterministic structure
        decay_factor = frequencies ** alpha
        amplitudes =np.random.uniform(0.9, 1.1, n_components) / decay_factor

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

    magnitude = np.abs(X)

    if len(magnitude) > 1:
        magnitude [1:] *= 2
        
    if return_phase:
        return magnitude , np.angle(X)
    else:
        return magnitude


# 3. Defining noise
def wavelength_in_samples(fs_cont, frequency):
    wavelength = fs_cont / frequency
    
    return (wavelength)
def low_freq_sin_noise(t, amplitude,
                                   frequency, target_power,
                                   T = 5.0, fs_cont = 10000):

    low_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_1 = np.mean(low_freq_sin**2)

    # Normalises_noise
    low_freq_sin *= np.sqrt(target_power / current_power_1)
    current_power_1 = np.mean(low_freq_sin ** 2)
    
    # Call function to work out wavelength
    wavelength_1 = wavelength_in_samples(fs_cont, frequency)

    return low_freq_sin, current_power_1, wavelength_1

def high_freq_sin_noise(t, amplitude,
                                   frequency, target_power,
                                   T = 5.0, fs_cont = 10000):
    
    high_freq_sin = amplitude * np.sin(2 * np.pi * frequency * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_2 = np.mean(high_freq_sin ** 2)
    # Normalises noise
    high_freq_sin *= np.sqrt(target_power / current_power_2)
    current_power_2 = np.mean(high_freq_sin ** 2)
    
    # Call function to work out wavelength
    wavelength_2 = wavelength_in_samples(fs_cont, frequency)

    return high_freq_sin, current_power_2, wavelength_2

def multi_tone_sin_noise(t, amplitude_min,
                                   amplitude_max,
                                   frequency_min,
                                   frequency_max, num_sin,
                                   target_power, T = 5.0, fs_cont = 10000):

    amplitudes = np.random.uniform(amplitude_min, amplitude_max, num_sin)
    frequencies = np.random.uniform(frequency_min, frequency_max, num_sin)

    # Build combined sine wave signal
    multi_tone_sin = np.zeros_like(t)
    for A, f in zip(amplitudes, frequencies):
        multi_tone_sin += A * np.sin(2 * np.pi * f * t + 0)

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_3 = np.mean(multi_tone_sin**2)
    # Normalises noise
    multi_tone_sin *= np.sqrt(target_power / current_power_3)
    current_power_3 = np.mean(multi_tone_sin ** 2)
    
    return multi_tone_sin, current_power_3

                           
# Acts as the control
def white_gaussian_noise(t, amplitude, target_power):
    white_noise = amplitude * np.random.randn(len(t))

    # Works out power, so that we can compare how closely it gets back to original power
    current_power_4 = np.mean(white_noise ** 2)

    # Normalises_noise
    white_noise *= np.sqrt(target_power / current_power_4)
    current_power_4 = np.mean(white_noise ** 2)

    return white_noise, current_power_4


# 4. Sampling
# Prime Modulus Sampling

# Co-Prime Sampling

# Random Prime Sampling

# Multiple Prime Step Sampling

# Uniform Sampling
def Uniform_Sampling(fs_sample, noisy_signal, t, option): # fs_sample input
    if fs_sample > fs_cont:
        raise ValueError("Sampling frequency cannot exceed continuous frequency")

    step = round(fs_cont / fs_sample)

    indicies = np.arrange(0,  len(t), step)
    
    sampled_signal = noisy_signal [indicies]
    sampled_t = t[indicies]

    return sampled_signal, sampled_t, indicies


# Random Sampling
def Random_Sampling(fs_sample, noisy_signal, t):
    if fs_sample > fs_cont:
        raise ValueError("Sampling frequency cannot exceed continuous frequency")

    n_samples = int(fs_sample * T)

    indicies = np.sort(np.random.choice(len(t), n_samples, replace = False))

    sampled_signal = noisy_signal[indicies]
    sampled_t = t[indicies] # In case we want to use this later to see where it sampled

    return sampled_signal, sampled_t, indicies
    

# Random interval sampling

# Jittered Uniform Sampling

# 5. Analysis

# 6. Matplotlib of Continuous signal and Magnitude spectrum
def graph_Csignal_Mspec (t, signal, frequency_grid, X):
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
    plt.plot(frequency_grid, X)
    plt.title("NUDFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0, 51, 1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 7. Matplotlib of Noisy signal and Noisy Magnitude spectrum overlaid with original
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
    plt.xticks(np.arange(0, 51, 1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 8. Matplotlib graph for sampled signal and nudft
def graph_Ssignal_Mspec (sampled_t, sampled_signal, t,
                                    noisy_signal, signal, frequency_grid, Y, Z, X):
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
    plt.plot(frequency_grid, Y, "b", label = "Noisy Magnitude Spectrum")
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
    plt.plot(frequency_grid, X, "b", label = "Original Magnitude Spectrum")
    plt.legend()
    plt.title("NUDFT Magnitude Spectrum (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Amplitude)")
    plt.xticks(np.arange(0,51,1))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main section
if __name__ == "__main__":

    # Parameters
    n_components = int(input("Please input the number of components for the sinusoid (recommended 8-20 for moderate complexity) (A realistic signal is 20-50):  "))
    target_power = float(input("Please input target power, recommended 1.0:  "))
    min_freq = float(input("Please input the minimum frequency, 1 / T = " + str(1/T) + ":  "))
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
    # Generate Signal
    t, signal, current_power = generate_signal(
        n_components, target_power,
        alpha, option, min_freq)

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
    print("The power level is currently", current_power)

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
                target_power = float(input("Please input target power, recommended 0.01:  "))
                
                noise, _ , wavelength_1 = low_freq_sin_noise(t, amplitude, frequency, target_power)
                noisy_signal += noise

                print("Wavelength is", round(wavelength_1, 3), "samples / wavelength")

            elif noise_opt == 1:
                amplitude = float(input("Please input an amplitude, recommended 0.2 - 0.4:  "))
                frequency = float(input("Please input a frequency, recommended 25 - 40 Hz:  "))
                target_power = float(input("Please input target power, recommended 0.01:  "))

                noise, _ , wavelength_2 = high_freq_sin_noise(t, amplitude, frequency, target_power)
                noisy_signal += noise

                print("Wavelength is", round(wavelength_2, 3), "samples / wavelength")

            elif noise_opt == 2:
                amplitude_min = float(input("Please input a minimum amplitude, recommended 0.2:  "))
                amplitude_max = float(input("Please input a maximum amplitude, recommended 0.4:  "))
                frequency_min = float(input("Please input a minimum frequency, recommended 25 Hz:  "))
                frequency_max = float(input("Please input a maximum frequency, recommended 40Hz:  "))
                num_sin = int(input("Please input the number of sin waves added as noise:  "))
                target_power = float(input("Please input target power, recommended 0.01:  "))

                noise, _ = multi_tone_sin_noise(t, amplitude_min,
                                                                amplitude_max,
                                                                frequency_min,
                                                                frequency_max, num_sin, target_power)
                noisy_signal += noise

                print("Wavelength can't be worked outer here (sum of multiple sine waves)")

            elif noise_opt == 3:
                # Recommended to be amplitude 0.2
                amplitude = float(input("Please input a float, recommended 0.2:  "))
                target_power = float(input("Please input target power, recommended 0.01:  "))
                
                noise, _ = white_gaussian_noise(t, amplitude, target_power)
                noisy_signal += noise

        except:
            raise ValueError("Must choose options 0-4")

    if np.array_equal(noisy_signal, signal):
        raise ValueError("Must choose one of options 0-3 before exiting")
    
    # Frequency grid for NUDFT
    frequency_grid = np.linspace(0, 50, 3000)

    # Compute Spectrum (called Y so can use X later on)
    Y = NUDFT_reconstruction(noisy_signal, t, frequency_grid)

    # Call Graph for noisy signal and Magnitude spectrum
    graph_noisy_vsignal (t, signal, noisy_signal, frequency_grid, X, Y)

    # Working out and printing the relative noise power
    relative_noise_power = np.mean((noisy_signal - signal) **2) / np.mean(signal ** 2)
    print("The relative noise power is", round(relative_noise_power, 5))

    # Working out the noise level from the power ratio of noisy signal to signal
    noise_db = 10 *np.log10(relative_noise_power)
    print("Relative noise level:", round(noise_db, 5), "dB")

    # Working out and printing the noise power and current signal power
    noise_power = np.mean((signal - noisy_signal) ** 2)
    print("The power in the noise is", round(noise_power, 5))
          
    noisy_signal_power = np.mean(noisy_signal ** 2)
    print("The current noisy signal power is", round(noisy_signal_power, 5))

    # Sampling
    noise_opt = 0
    while noise_opt != 4:
        fs_sample = int(input(("Please input the sample rate for uniform, MUST BE > 2 * max frequency and < " + str(fs_cont) +  " :  ")))
        noise_opt = int(input("Input noise you want, (4 stops sampling) (0 =  Uniform sampling) (1 =  Random sampling):  ") )
        
        try:
            if noise_opt == 0:
                 sampled_signal, sampled_t, indicies = Uniform_Sampling(fs_sample, noisy_signal, t)

            elif noise_opt == 1:
                sampled_signal, sampled_t, indicies = Random_Sampling(fs_sample, noisy_signal, t)

            elif option == 4:
                break
            
        except:
            raise ValueError("Must choose options 0 - 4")

   
        frequency_grid = np.linspace(0, 50, 3000)
        Z = NUDFT_reconstruction(sampled_signal, sampled_t, frequency_grid)
        graph_Ssignal_Mspec (sampled_t, sampled_signal, t,
                                        noisy_signal, signal, frequency_grid, Y, Z, X)

        # Work out the sampled noise and percentage reduction

        sampled_power = np.mean(sampled_signal ** 2)
        print("The sampled_power is", round(sampled_power, 5))


        sampled_noise_power = np.mean((signal[indicies] - sampled_signal) ** 2)
        print("The power in the noise is", round(sampled_noise_power, 5))

        percent_reduction_noise = sampled_noise_power / noise_power
        print("The percentage reduction is", round(percent_reduction_noise, 5))


