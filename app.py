import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# -----------------------------
# --- Helper: Bandpass Filter ---
# -----------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# -----------------------------
# --- Load Excel & Prepare Subjects ---
# -----------------------------
st.title("Interactive BCG + ECG Viewer (Single Excel)")

# Directly read Main_excel.xlsx from repo
excel_file = "Main_excel.xlsx"
xl = pd.ExcelFile(excel_file)
sheets = xl.sheet_names

# Extract unique subjects based on sheet naming: e.g., "sub1_BCG" -> "sub1"
subjects = sorted(list(set([s.split('_')[0] for s in sheets])))
subject = st.selectbox("Select Subject", subjects)

# Load BCG & ECG sheets for selected subject
df_bcg = pd.read_excel(excel_file, sheet_name=f"{subject}_BCG")
df_ecg = pd.read_excel(excel_file, sheet_name=f"{subject}_ECG")

bcg = df_bcg['BCG'].values
t_bcg = df_bcg['Time_BCG'].values
ecg = df_ecg['ECG'].values
t_ecg = df_ecg['Time_ECG'].values

# -----------------------------
# --- Interactive sliders for BCG ---
# -----------------------------
st.subheader("Band-Pass Filter for BCG")
lowcut = st.slider("Low Cutoff Frequency (Hz)", 0.1, 10.0, 0.5, 0.01)
highcut = st.slider("High Cutoff Frequency (Hz)", 1.0, 60.0, 5.0, 0.01)

if lowcut >= highcut:
    st.warning("⚠️ Low cutoff must be smaller than high cutoff")
    bcg_filtered = bcg
else:
    fs_bcg = 1 / np.mean(np.diff(t_bcg))  # BCG sampling rate
    bcg_filtered = bandpass_filter(bcg, lowcut, highcut, fs_bcg)

# -----------------------------
# --- Interpolate BCG to ECG time ---
# -----------------------------
interp_bcg = interp1d(t_bcg, bcg_filtered, kind='linear', bounds_error=False, fill_value="extrapolate")
bcg_resampled = interp_bcg(t_ecg)

# -----------------------------
# --- Plot overlay ---
# -----------------------------
st.subheader("Signals Overlay")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_ecg, ecg, label='ECG', color='red')
ax.plot(t_ecg, bcg_resampled, label='Filtered BCG', color='blue')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude")
ax.legend()
st.pyplot(fig)
