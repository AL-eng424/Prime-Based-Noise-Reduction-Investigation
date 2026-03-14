# Signal Sampling and Noise Analysis

An experimental project investigating how **different sampling strategies interact with periodic noise** in signals.

The repository generates synthetic signals, injects controllable noise, samples the signal, and analyses the result using a **Non-Uniform Discrete Fourier Transform (NUDFT)**. The aim is to explore how sampling patterns influence noise behaviour and signal reconstruction.

**Contributors**

* **AL-eng424** – Prime / modular sampling experiment
* **WC-eng424** – Signal generation, noise modelling, and NUDFT analysis

---

# Repository Structure

The project is currently split into **two independent prototype scripts**.
They explore different parts of the signal-processing pipeline and will be integrated later.

```text
final_NUDFT.py
Prime_Average_Sampler.py
```

Both scripts can be run directly and guide the user through experiments using terminal prompts.

---

# final_NUDFT.py — Signal Generation and Spectral Analysis

*(WC-eng424)*

This script provides a small environment for experimenting with signals and noise.

Main features:

* Synthetic signal generation using a sum of sinusoidal components
* Configurable noise injection
* Uniform sampling of the signal
* Spectral analysis using a **Non-Uniform Discrete Fourier Transform (NUDFT)**

Two signal models are available:

* **Flat amplitudes** – random sinusoidal amplitudes
* **Spectral decay** – amplitudes proportional to (1 / f^{\alpha})

Several noise types can be added interactively:

* Low-frequency sinusoidal noise
* High-frequency sinusoidal noise
* Multi-tone sinusoidal noise
* White Gaussian noise

The script visualises:

* the original signal
* the noisy signal
* sampled signals
* magnitude spectra from the NUDFT

Run with:

```bash
python final_NUDFT.py
```

---

# Prime_Average_Sampler.py — Modular Sampling Experiment

*(AL-eng424)*

This script explores an experimental sampling method aimed at reducing periodic noise through **repeated modular sampling passes**.

The idea is that periodic noise can sometimes align with a fixed sampling interval. If the same noise phase is repeatedly sampled, averaging multiple reconstructions may fail to cancel the noise.

The algorithm attempts to vary the sampling phase by:

1. Sampling the noisy signal with interval `k`
2. Reconstructing the waveform using interpolation
3. Shifting the starting sample using modular arithmetic
4. Repeating several sampling passes
5. Averaging the reconstructed waveforms

Example parameters:

```text
N        = signal length
k        = sampling interval
T_signal = signal period
T_noise  = noise period
```

The script plots the **clean signal and averaged reconstruction** for comparison.

Run with:

```bash
python Prime_Average_Sampler.py
```

---

# Concept

A noisy signal can be written as:

```
observed signal = true signal + periodic noise
```

If the sampling interval aligns with the noise period, the same noise phase may be sampled repeatedly. In theory, using sampling parameters that are **coprime** can distribute the sampled phases across the noise cycle.

The modular offset update used in the experiment is:

```
j = (j + num_samples_taken * k) mod N
```

This allows repeated sampling passes while keeping indices within the signal window.

---

# Current Status

The repository currently contains **two separate prototypes**:

* signal generation and spectral analysis (`final_NUDFT.py`)
* experimental modular sampling (`Prime_Average_Sampler.py`)

They are not yet integrated but can be run independently to demonstrate each part of the project.

Future work may include integrating the sampling algorithm into the NUDFT pipeline and exploring additional sampling strategies.

---

# Requirements

Python packages used in this project:

```
numpy
matplotlib
sympy
```
