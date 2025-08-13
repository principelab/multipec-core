# Usage

## Run in sequence:

## 1. Preprocessing

```bash
python -m multipec.eeg_preprocess --input data/input/eeg --output data/preprocessed/eeg/
```

## 2. MultiPEC

```bash
python -m multipec.multipec_nets --load_path data/preprocessed/eeg/ --save_path data/output/eeg/
```