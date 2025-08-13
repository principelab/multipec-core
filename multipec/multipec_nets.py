import os
import argparse
from copy import deepcopy
from pickle import load, dump
import numpy as np
from simulation_utils import PEC_pairs, direct, multipec

def run_multipec_analysis(file, path_load_data, path_save_data):
    # Detect if CNN (only one file) or EEG (multiple files)
    if "eeg" in path_load_data.lower():
        stim_type_str, sub, *_ = file.replace(".npy", "").split("_")
        filename_ref = f"{stim_type_str}_{sub}"
    elif "cnn" in path_load_data.lower():
        filename_ref = "cnn"
    else:
        raise ValueError("Unknown data source. 'cnn' or 'eeg' should be part of the input path.")

    print(f"\nProcessing {file}")
    signal_bin = np.load(os.path.join(path_load_data, file), allow_pickle=True).item()

    # Load or compute PEC pairs
    pairs_path = os.path.join(path_save_data, f"pairs_{filename_ref}.p")
    if os.path.isfile(pairs_path):
        pairs = load(open(pairs_path, "rb"))
    else:
        pairs = PEC_pairs(signal_bin)
        dump(pairs, open(pairs_path, "wb"))

    # Direct pairs and sort them by error
    nodes = list(signal_bin.keys())
    directed_pairs = direct(pairs, nodes)
    sorted_pairs = dict(sorted(directed_pairs.items(), key=lambda item: item[1]))

    # Error thresholds
    median_err = np.median(list(sorted_pairs.values()))
    std_err = np.std(list(sorted_pairs.values()))

    # Define all pair categories to compute
    pair_sets = {
        "nets_4down": {s: v for s, v in sorted_pairs.items() if v < median_err - 4 * std_err},
        "nets_34": {s: v for s, v in sorted_pairs.items() if median_err - 4 * std_err < v < median_err - 3 * std_err},
        "nets_23": {s: v for s, v in sorted_pairs.items() if median_err - 3 * std_err < v < median_err - 2 * std_err},
        "nets_12": {s: v for s, v in sorted_pairs.items() if median_err - 2 * std_err < v < median_err - 1 * std_err},
        "nets_05median": {s: v for s, v in sorted_pairs.items() if median_err - 0.05 * std_err < v < median_err + 0.05 * std_err},
        "nets_05up": {s: v for s, v in sorted_pairs.items() if v > median_err + 0.5 * std_err}
    }

    if path_load_data == "path/preprocessed/eeg/":
        # EEG-specific processing
        pair_sets = {
            "nets_3down": {s: v for s, v in sorted_pairs.items() if v < median_err - 3 * std_err}
        }

    # Shared setup
    labels = list(signal_bin.keys())
    node_signal_bin = {node: signal_bin[node] for node in labels}

    for label, take_pairs in pair_sets.items():
        if not take_pairs:
            continue

        nonselected = deepcopy(labels)
        nets = multipec(node_signal_bin, labels, take_pairs, nonselected, n_nets=len(take_pairs))
        nets_path = os.path.join(path_save_data, f"{label}_{filename_ref}.p")
        dump(nets, open(nets_path, "wb"))


if __name__ == "__main__":
    # Input for EEG: data/preprocessed/eeg/
    # Output for EEG: data/output/eeg/
    # Input for CNN: data/preprocessed/cnn/
    # Output for CNN: data/output/cnn/
    # Example: python multipec_nets.py --load_path data/preprocessed/eeg/ --save_path data/output/eeg/
    parser = argparse.ArgumentParser(description="Run MultiPEC")
    parser.add_argument("--load_path", required=True, help="Path to directory containing preprocessed files")
    parser.add_argument("--save_path", required=True, help="Path to save analysis results")

    args = parser.parse_args()

    file_list = [f for f in os.listdir(args.load_path) if f.endswith('_preprocessed.npy')]
    for file in file_list:
        try:
            run_multipec_analysis(file, args.load_path, args.save_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
