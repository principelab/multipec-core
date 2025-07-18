import os
import argparse
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from scipy.signal import medfilt
import warnings
from data_legacy import binarize

def bandpass_filter(data, fs, lowcut, highcut):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=1)

def notch_filter(data, fs, notch_freq):
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist
    b, a = signal.iirnotch(notch, 30, fs)
    return signal.filtfilt(b, a, data, axis=1)

def preprocess_eeg(file, path_load_data, path_save_data):
    sub = file.split(".")[0].split("_")[-1]
    print(f"\nProcessing Subject #{sub}")

    eeg_path = os.path.join(path_load_data, file.replace(".vhdr", ".eeg"))
    eeg_data = np.fromfile(eeg_path, dtype=np.float32).reshape((64, -1), order="F")

    eeg_data = bandpass_filter(eeg_data, 500, 1, 100)
    eeg_data = notch_filter(eeg_data, 500, 50)
    eeg_data = signal.resample(eeg_data, int(eeg_data.shape[1] * (250 / 500)), axis=1)
    new_fs = 250

    vmrk_path = os.path.join(path_load_data, file.replace(".vhdr", ".vmrk"))
    event_markers = []
    with open(vmrk_path, "r") as f:
        for line in f:
            if line.startswith("Mk"):
                parts = line.strip().split(",")
                event_markers.append((parts[1], int(parts[2])))

    stim_types = ['S  1', 'S  2', 'S  3', 'S  4', 'S  5', 'S  6', 'S  7']
    for stim_type in stim_types:
        stim_type_str = stim_type.replace(" ", "")
        print(f"  Stimulus: {stim_type_str}")

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                epochs = []
                window_samples = 30 * new_fs
                for event, sample in event_markers:
                    if event == stim_type and 0 <= sample < eeg_data.shape[1] - window_samples:
                        epochs.append(eeg_data[:, sample:sample + window_samples])

                if not epochs:
                    continue

                epochs = np.hstack(epochs)
                epochs -= np.mean(epochs, axis=0, keepdims=True)

                z_thresh = 5
                mean_per_channel = np.mean(epochs, axis=1, keepdims=True)
                std_per_channel = np.std(epochs, axis=1, keepdims=True)
                z_scores = np.abs((epochs - mean_per_channel) / std_per_channel)

                artifact_mask = z_scores > z_thresh
                epochs_cleaned = np.copy(epochs)
                for ch in range(epochs.shape[0]):
                    if np.any(artifact_mask[ch]):
                        epochs_cleaned[ch, artifact_mask[ch]] = medfilt(epochs[ch, artifact_mask[ch]], kernel_size=5)

                ica = FastICA(n_components=epochs_cleaned.shape[0], random_state=0, max_iter=3000, tol=1e-6)
                components = ica.fit_transform(epochs_cleaned.T).T

                artifact_components = np.where(kurtosis(components, axis=1) > 5)[0]
                components[artifact_components, :] = 0
                epochs_cleaned = ica.inverse_transform(components.T).T

                A1_idx, A2_idx = 31, 32
                if A1_idx < epochs_cleaned.shape[0] and A2_idx < epochs_cleaned.shape[0]:
                    mastoid_avg = np.mean(epochs_cleaned[[A1_idx, A2_idx], :], axis=0, keepdims=True)
                    epochs_cleaned -= mastoid_avg
                else:
                    print(f"  Skipping mastoid re-referencing (invalid channel indices)")

                epochs_cleaned = np.delete(epochs_cleaned, [A1_idx, A2_idx], axis=0)

                skip_due_to_warning = any(
                    "FastICA did not converge" in str(warn.message) or 
                    "kernel_size exceeds volume extent" in str(warn.message)
                    for warn in w
                )

                if skip_due_to_warning:
                    raise RuntimeError("Warning triggered skip")

                # Binarize the signal
                signal, nodes = epochs_cleaned['data'], epochs_cleaned['nodes']
                signal_bin = {id_node:binarize(arr_node) for id_node, arr_node in enumerate(signal)}
                # Remove the mastoid channels if they exist
                for key in [31, 32]:
                    signal_bin.pop(key, None)

                # Save the preprocessed binarized data
                save_path = os.path.join(path_save_data, f"{stim_type_str}_{sub}_preprocessed.npy")
                np.save(save_path, signal_bin)

        except Exception as e:
            print(f"  Skipping {stim_type_str} due to warning or error: {e}")
            continue

def preprocess_all_files(path_load_data, path_save_data):
    os.makedirs(path_save_data, exist_ok=True)

    vhdr_files = [f for f in os.listdir(path_load_data) if f.endswith(".vhdr")]
    if not vhdr_files:
        raise FileNotFoundError("No .vhdr files found in input directory.")

    for file in vhdr_files:
        preprocess_eeg(file, path_load_data, path_save_data)

if __name__ == "__main__":
    # Input for EEG: data/input/eeg/
    # Output for EEG: data/preprocessed/eeg/
    parser = argparse.ArgumentParser(description="Preprocess EEG .vhdr files.")
    parser.add_argument("--input", required=True, help="Path to the folder containing .vhdr EEG files.")
    parser.add_argument("--output", required=True, help="Path to save the preprocessed files.")
    
    args = parser.parse_args()

    preprocess_all_files(args.input, args.output)
