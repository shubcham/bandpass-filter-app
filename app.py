import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import plotly.graph_objects as go

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
# --- Normalization Helper (log-scale for visualization) ---
# -----------------------------
def log_scale(sig):
    return np.sign(sig) * np.log1p(np.abs(sig))

# -----------------------------
# --- Load Excel & Prepare Subjects ---
# -----------------------------
st.title("Interactive BCG + ECG Viewer (Single Excel)")

excel_file = "Main_excel.xlsx"
xl = pd.ExcelFile(excel_file)
sheets = xl.sheet_names

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
    hr_values = df_hr.to_numpy().flatten()
    hr_values = [v for v in hr_values if pd.notna(v) and isinstance(v, (int, float))]
    hr_value = hr_values[0] if len(hr_values) > 0 else np.nan
except Exception as e:
    st.warning(f"HR sheet not found or invalid format: {e}")
    hr_value = np.nan

# HR display
if isinstance(hr_value, (int, float)) and not np.isnan(hr_value):
    st.write(f"**Average Heart Rate:** {hr_value:.2f} BPM ({hr_value/60:.3f} Hz)")
    ecg_label = f'ECG, HR={hr_value/60:.3f} Hz'
else:
    st.write("**Average Heart Rate:** Not available")
    ecg_label = 'ECG, HR=NA'

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
    fs_bcg = 1 / np.mean(np.diff(t_bcg))
    bcg_filtered = bandpass_filter(bcg, lowcut, highcut, fs_bcg)

# -----------------------------
# --- Interpolate BCG to ECG time ---
# -----------------------------
interp_bcg = interp1d(t_bcg, bcg_filtered, kind='linear', bounds_error=False, fill_value="extrapolate")
bcg_resampled = interp_bcg(t_ecg)

# -----------------------------
# --- Log scale for visualization ---
# -----------------------------
bcg_vis = log_scale(bcg_resampled)
ecg_vis = log_scale(ecg)

# -----------------------------
# --- Plot using Plotly for interactive zoom ---
# -----------------------------
st.subheader("Signals Overlay (Interactive)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_ecg, y=ecg_vis, mode='lines', name=ecg_label, line=dict(color='red'), opacity=0.5))
fig.add_trace(go.Scatter(x=t_ecg, y=bcg_vis, mode='lines', name='Filtered BCG', line=dict(color='blue')))

fig.update_layout(
    title="ECG and BCG Overlay",
    xaxis_title="Time [s]",
    yaxis_title="Amplitude (log-scaled)",
    hovermode='x unified'
)

# Enable zooming, panning, and reset axes
fig.update_xaxes(rangeslider_visible=True)  # optional: mini range slider
fig.update_yaxes(fixedrange=False)

st.plotly_chart(fig, use_container_width=True)
