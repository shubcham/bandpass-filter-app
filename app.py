import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# Set page configuration for wide mode
st.set_page_config(layout="wide")

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
st.title("Interactive BCG + ECG Viewer")

excel_file = "Main_excel.xlsx"
xl = pd.ExcelFile(excel_file)
sheets = xl.sheet_names

subjects = sorted(list(set([s.split('_')[0] for s in sheets])))
st.markdown("<h3 style='font-weight:bold'>Select Subject</h3>", unsafe_allow_html=True)
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
# --- Interactive sliders & manual input for BCG ---
# -----------------------------
st.subheader("Band-Pass Filter for BCG")

# Create two columns: slider + manual input
col1, col2 = st.columns(2)

with col1:
    lowcut_slider = st.slider("Low Cutoff Frequency (Hz)", 0.1, 4.0, 0.5, 0.01)
with col2:
    lowcut_input = st.number_input("Low Cutoff Manual Input (Hz)", min_value=0.01, max_value=10.0, value=0.5, step=0.01)

with col1:
    highcut_slider = st.slider("High Cutoff Frequency (Hz)", 1.0, 7.0, 2.0, 0.01)
with col2:
    highcut_input = st.number_input("High Cutoff Manual Input (Hz)", min_value=0.1, max_value=60.0, value=5.0, step=0.01)

# Decide which values to use (manual input overrides slider if changed)
lowcut = lowcut_input if lowcut_input != 0.5 else lowcut_slider
highcut = highcut_input if highcut_input != 2.0 else highcut_slider

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
fig.add_trace(go.Scatter(
    x=t_ecg, y=ecg_vis, mode='lines', name=ecg_label,
    line=dict(color='red'), opacity=0.5
))
fig.add_trace(go.Scatter(
    x=t_ecg, y=bcg_vis, mode='lines', name='Filtered BCG',
    line=dict(color='blue')
))

fig.update_layout(
    title="ECG and BCG Overlay",
    title_font=dict(size=24, family="Arial", color="black"),  # Bigger title
    xaxis_title="Time [s]",
    yaxis_title="Amplitude (log-scaled)",
    xaxis=dict(title_font=dict(size=20), tickfont=dict(size=20)),  # Bigger x-axis labels
    yaxis=dict(title_font=dict(size=20), tickfont=dict(size=20)),  # Bigger y-axis labels
    legend=dict(font=dict(size=20)),  # Bigger legend font
    width=900,
    height=600,
    margin=dict(l=40, r=40, t=60, b=40),
    hovermode='x unified'
)

fig.update_xaxes(fixedrange=False)
fig.update_yaxes(fixedrange=False)

st.plotly_chart(fig, use_container_width=True)
