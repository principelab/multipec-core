# Usage

## Run in sequence:

## 1. Activations

```bash
python -m multipec.cnn_activation
```

## 2. Preprocessing

```bash
python -m multipec.cnn_preprocess --input data/preprocessed/cnn/ --output data/output/cnn/
```

## 3. MultiPEC

```bash
python -m multipec.multipec_nets --load_path data/preprocessed/cnn/ --save_path data/output/cnn/
```


