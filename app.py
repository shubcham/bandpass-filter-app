import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from io import BytesIO

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

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file with all subjects", type=["xlsx"])
if uploaded_file is not None:
    # Read all sheet names
    xl = pd.ExcelFile(uploaded_file)
    sheets = xl.sheet_names

    # Extract unique subjects based on sheet naming: e.g., "sub1_BCG" -> "sub1"
    subjects = sorted(list(set([s.split('_')[0] for s in sheets])))
    subject = st.selectbox("Select Subject", subjects)

    # Load BCG & ECG sheets for selected subject
    df_bcg = pd.read_excel(uploaded_file, sheet_name=f"{subject}_BCG")
    df_ecg = pd.read_excel(uploaded_file, sheet_name=f"{subject}_ECG")

    bcg = df_bcg['BCG'].values
    t_bcg = df_bcg['Time_BCG'].values
    ecg = df_ecg['ECG'].values
    t_ecg = df_ecg['Time_ECG'].values

    # -----------------------------
    # --- Interactive sliders for BCG ---
    # -----------------------------
    st.subheader("Band-Pass Filter for BCG")
    lowcut = st.slider("Low Cutoff Frequency (Hz)", 0.1, 10.0, 0.5, 0.1)
    highcut = st.slider("High Cutoff Frequency (Hz)", 1.0, 60.0, 5.0, 0.5)

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

    # # -----------------------------
    # # --- Optional: download filtered BCG ---
    # # -----------------------------
    # df_filtered = pd.DataFrame({"Time": t_bcg, "BCG_filtered": bcg_filtered})
    # def to_excel(df):
    #     output = BytesIO()
    #     with pd.ExcelWriter(output, engine='openpyxl') as writer:
    #         df.to_excel(writer, index=False, sheet_name='BCG_filtered')
    #     return output.getvalue()

    # excel_data = to_excel(df_filtered)
    # st.download_button(
    #     label="📥 Download Filtered BCG",
    #     data=excel_data,
    #     file_name=f"{subject}_BCG_filtered.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    # )
