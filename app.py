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
# --- Normalization Helper ---
# -----------------------------
def normalize_signal(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))

# -----------------------------
# --- Load Excel & Prepare Subjects ---
# -----------------------------
st.title("Interactive BCG + ECG Viewer (Single Excel)")

# Read Main_excel.xlsx from repo
excel_file = "Main_excel.xlsx"
xl = pd.ExcelFile(excel_file)
sheets = xl.sheet_names

# Extract unique subjects based on sheet naming, e.g., "sub1_BCG" -> "sub1"
subjects = sorted(list(set([s.split('_')[0] for s in sheets])))
subject = st.selectbox("Select Subject", subjects)

# -----------------------------
# --- Load BCG & ECG Sheets ---
# -----------------------------
df_bcg = pd.read_excel(excel_file, sheet_name=f"{subject}_BCG")
df_ecg = pd.read_excel(excel_file, sheet_name=f"{subject}_ECG")

bcg = df_bcg['BCG'].values
t_bcg = df_bcg['Time_BCG'].values
ecg = df_ecg['ECG'].values
t_ecg = df_ecg['Time_ECG'].values

# -----------------------------
# --- Load HR from HR Sheet ---
# -----------------------------
try:
    df_hr = pd.read_excel(excel_file, sheet_name=f"{subject}_HR", header=None)
    # Flatten to 1D array and take first numeric value
    hr_values = df_hr.to_numpy().flatten()
    hr_values = [v for v in hr_values if pd.notna(v)]
    if len(hr_values) > 0:
        hr_value = hr_values[0]
    else:
        hr_value = np.nan
except Exception as e:
    st.warning(f"HR sheet not found or invalid format: {e}")
    hr_value = np.nan

# -----------------------------
# --- Display HR safely ---
# -----------------------------
if isinstance(hr_value, (int, float)) and not np.isnan(hr_value):
    st.write(f"**Average Heart Rate:** {hr_value:.2f} Hz")
else:
    st.write("**Average Heart Rate:** Not available")


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
# --- Normalize AFTER filtering ---
# -----------------------------
bcg_norm = normalize_signal(bcg_resampled)
ecg_norm = normalize_signal(ecg)

# -----------------------------
# --- Plot overlay with HR in legend ---
# -----------------------------
st.subheader("Signals Overlay")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_ecg, ecg_norm, label=f'ECG, HR={hr_value:.2f} Hz', color='red', alpha=0.5)
ax.plot(t_ecg, bcg_norm, label='Filtered BCG', color='blue')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Normalized Amplitude (0-1)")
ax.legend()
st.pyplot(fig)
