# Modules Overview

### `simulation_utils.py`
Contains functions that support the simulates binary-encoded activity for virtual networks: helper functions for random stimulus creation, flattening nested data, binary conversion, and configuring binary signals to networks (two-network or three-network setups).
Implements MultiPEC.

### `data.py`, `data_legacy.py`, `core.py`, `core_legacy.py`
Structures for data wrangling and storage.

### `awc.py`
Implements a context-based, bit-level learning system (Average Weighted Context) to model and predict sequences, computing reconstruction errors (error) between data series or nodes. The AWC is the basis of PEC.

### `eeg_preprocess.py`
Loads raw EEG files, applies bandpass and notch filtering, resamples the data, segments it into stimulus-specific epochs, removes artifacts using z-score thresholding and ICA.
Converts cleaned EEG signals into binarized form per channel (excluding mastoids), and saves the preprocessed data for MultiPEC analysis.

### `cnn_train.py`
MNIST classification using a CNN, using SGD with momentum and negative log-likelihood loss. Tracks and saves training and test losses, and periodically saves model and optimizer states.

### `cnn_activation.py`
Few-shot training of a pre-trained CNN on MNIST. Selects a fixed number of images per class, trains the model for a few epochs, and saves per-batch weights and intermediate layer outputs using forward hooks.
Evaluates on the full MNIST test set each epoch, and saves performance metrics to an Excel file `task_metrics.xlsx`.

### `cnn_preprocess.py`
Extracts and processes CNN layer activations. Loads saved per-batch outputs of convolutional layers, computes the mean activation per feature map, and organizes them into a series signal array across epochs.
Binarizes and saves preprocessed activations as a .npy file for MultiPEC analysis.

### `cnn_pruning.py`
Evaluates networks identified by MultiPEC, by pruning. Loads a pre-trained CNN, applies masks to prune specific neurons in convolutional layers according to predefined networks, and computes performance metrics for the overall model and each class.
Stores results in `data/results/cnn/` as Excel files `pruned_*.xlsx`.

### `cnn_pruning_random.py`
Loads a pre-trained CNN, prunes 1,000 random subsets of its convolutional filters, and evaluates overall and per-class accuracy, F1, precision, and recall to track the effect of pruning.
Measures class specificity, logs detailed results per iteration and saves the log in `data/results/cnn/` as Excel files `global_random_pruning_log.xlsx`.

### `multipec_nets.py`
Entry point for running the entire MultiPEC analysis.

