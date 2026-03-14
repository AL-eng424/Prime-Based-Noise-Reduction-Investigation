import numpy as np
import matplotlib.pyplot as plt

# Requirements: T_noise < k < T_signal < N
# For optimum performance, N, k and T_noise should be coprime.

N = int(input('Input signal length N (recommended: prime number around 111): '))
k = int(input('Input sample interval k (recommended: prime number around 19): '))
T_signal = int(input('Input signal period T_signal (recommended: N > T_signal > k, E.G. 29): '))
T_noise = int(input('Input noise period T_noise (recommended: N > T_signal > k > T_noise, E.G. 16): '))

t = np.linspace(0, N, N, endpoint=False)
signal = np.sin(t * 2 * np.pi / T_signal)
noise = np.sin(t * 2 * np.pi / T_noise)
noisy_signal = signal + noise

plt.figure(figsize=(10, 4))

# Sampling logic
j = 0 # k offset
avg_waveform = np.zeros_like(t)
num_passes = 111 # Instead of k < k*N, we set a finite number of passes

for n in range(1, num_passes + 1):
    # 1. Grab samples starting at the current offset j
    sampled_t = t[j::k]
    sampled_noisy_signal = noisy_signal[j::k]
    
    # 2. Interpolate (period=N handles the wrap-around points)
    sampled_waveform = np.interp(t, sampled_t, sampled_noisy_signal, period=N)
    avg_waveform += sampled_waveform
    
    # 3. Wrapping logic: Find the NEXT starting index
    # We find where the next sample would have landed if N were infinite, 
    # then wrap that back into the range [0, N-1]
    num_samples_taken = len(sampled_t)
    j = (j + num_samples_taken * k) % N

# Final Average
avg_waveform /= num_passes

plt.plot(t, signal, label='Clean Signal')
plt.plot(t, avg_waveform, label='Averaged Reconstruction')
plt.legend()
plt.show()
