import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -----------------------------
# Helper: bandpass filter
# -----------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Interactive Band-Pass Filter Demo")

# Simulated raw signal (replace with file upload if needed)
fs = 1000  # sampling rate
t = np.linspace(0, 2, 2*fs, endpoint=False)
raw_signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)

# Sliders for cutoff frequencies (all floats)
lowcut = st.slider("Low Cutoff Frequency (Hz)", 0.5, 5.0, 1.0, 0.1)
highcut = st.slider("High Cutoff Frequency (Hz)", 5.0, 100.0, 20.0, 0.5)

# Apply band-pass filter
if lowcut < highcut:
    filtered_signal = bandpass_filter(raw_signal, lowcut, highcut, fs)
else:
    st.warning("⚠️ Low cutoff must be smaller than high cutoff")
    filtered_signal = raw_signal

# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t, raw_signal, color='gray')
ax[0].set_title("Raw Signal")
ax[1].plot(t, filtered_signal, color='blue')
ax[1].set_title(f"Filtered Signal ({lowcut}-{highcut} Hz)")
st.pyplot(fig)
