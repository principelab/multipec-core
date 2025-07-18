import os
import argparse
import numpy as np
from glob import glob
from data_legacy import binarize
import torch

# Helper: sort files by batch number
def batch_sort(filename):
    return int(filename.split("_")[1][1:])

def process_activations(input_folder, output_folder, task_len=1000, n_epochs=5):
    activations_folder = os.path.join(input_folder, "activations")
    os.makedirs(activations_folder, exist_ok=True)
    preprocessed_folder = output_folder
    os.makedirs(preprocessed_folder, exist_ok=True)

    path_model = os.path.join(input_folder, "model.pth")
    conv_layers = {2: 10, 3: 20}  # layer indices : number of filters; see Net() class in multipec_nets.py

    # Save layer activations
    for ilayer in conv_layers.keys():
        layer_folder = os.path.join(activations_folder, f"layer{ilayer}")
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder, exist_ok=True)
            for epoch in range(n_epochs):
                files_activations = glob(os.path.join(activations_folder, f"E{epoch}_*_outputs.pth"))

                for file in files_activations:
                    filename = os.path.basename(file)
                    E, B, stim = filename.split("_")[:3]
                    state_dict = torch.load(file, map_location="cpu")

                    for key, val in state_dict.items():
                        if key.split("_")[0] == str(ilayer):
                            act = val.detach().cpu().numpy()
                            np.save(os.path.join(activations_folder, f"{E}_{B}_{stim}_output_layer{ilayer}"), act)

    signal = np.zeros(shape=(sum(conv_layers.values()), task_len))
    t_start = 0

    for epoch in range(n_epochs):
        l2_files = glob(os.path.join(activations_folder, "layer2", f"E{epoch}_*_layer2.npy"))
        l3_files = glob(os.path.join(activations_folder, "layer3", f"E{epoch}_*_layer3.npy"))

        l2_files.sort(key=batch_sort)
        l3_files.sort(key=batch_sort)

        for t2, filename2 in enumerate(l2_files):
            output2 = np.load(filename2)
            for fmap_id, fmap in enumerate(output2):
                signal[fmap_id][t_start + t2] = fmap.mean()

        for t3, filename3 in enumerate(l3_files):
            output3 = np.load(filename3)
            for fmap_id, fmap in enumerate(output3):
                signal[conv_layers[2] + fmap_id][t_start + t3] = fmap.mean()

        t_start += len(l2_files)

    # Check for null (all-zero) arrays
    null_indices = [j for j, a in enumerate(signal) if np.all(a == 0.0)]
    if null_indices:
        raise ValueError(f"Null (all-zero) arrays found at indices: {null_indices}. Aborting process.")

    # Binarize signal and remove useless nodes
    signal_bin = {id_node: binarize(arr_node) for id_node, arr_node in enumerate(signal)}
    signal_bin = {k: v for k, v in signal_bin.items() if not np.all(v == 1)}

    # Save preprocessed binarized activations
    save_path = os.path.join(preprocessed_folder, "CNN_activations_preprocessed.npy")
    np.save(save_path, signal_bin)


if __name__ == "__main__":
    # Input for CNN: data/input/cnn/
    # Output for CNN: data/preprocessed/cnn/
    # Example usage:
    # python cnn_preprocess.py --input data/input/cnn/ --output data/preprocessed
    parser = argparse.ArgumentParser(description="Process CNN layer activations into binarized signals.")
    parser.add_argument("--input", required=True, help="Input folder containing CNN model and outputs.")
    parser.add_argument("--output", required=True, help="Folder where intermediate outputs are stored.")
    parser.add_argument("--save", required=True, help="Final save path for the binarized result.")

    args = parser.parse_args()

    process_activations(args.input, args.output, args.save)
