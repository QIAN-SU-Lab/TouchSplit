# Reproducing the paper results

## Data

Download the training datasets of DIC-C2DH-HeLa, Fluo-N2DL-HeLa, Fluo-C2DL-MSC and PhC-C2DH-U373 from https://celltrackingchallenge.net/ into one folder and set `data.ctc_root` in `touchsplit.yml`, keeping the CTC layout (`<dataset>/01`, `01_GT`, `01_ST`, `02`, ...).

- Training: all frames of sequence 01 (GT where available, ST elsewhere); one-pixel gaps between touching instances are filled.
- Testing: all frames of sequence 02 (GT where available, ST elsewhere).

## Pipeline

```bash
DS=DIC-C2DH-HeLa

# 1. Seed 0: fine-tune CellposeSAM (also the CPSAM baseline), then train the decoder
python touchsplit.py train -d $DS -s 0

# 2. Seeds 1 and 2: train the decoder on the seed-0 encoder
for SEED in 1 2; do
  python touchsplit.py train -d $DS -s $SEED -e weights/cellpose_ft_${DS}_s0.pt
done

# 3. Select the marker parameter on the training frames (paper values: decode in touchsplit.yml)
python touchsplit.py select -d $DS -e weights/cellpose_ft_${DS}_s0.pt -w weights/touchsplit_head_${DS}_s{0,1,2}.pt

# 4. Predict sequence 02 and evaluate (SEG, Touch SEG, Nontouch SEG)
for SEED in 0 1 2; do
  python touchsplit.py predict -d $DS -e weights/cellpose_ft_${DS}_s0.pt -w weights/touchsplit_head_${DS}_s${SEED}.pt \
      -i /path/to/CTC/$DS/02 -o predictions/${DS}_s${SEED}
done
python touchsplit.py evaluate -d $DS --pred predictions/${DS}_s{0,1,2}
```

CPSAM baseline seeds 1 and 2: `python touchsplit.py train -d $DS -s 1 --encoder-only`.

Platelet settings (data not public): decoder `lam: 2`, `lr: 1.0e-3`, 100 epochs, marker `md:16`, 8-fold dihedral test-time augmentation.

## Training time

Encoder fine-tuning takes 10 -- 30 min and decoder training 40 min -- 3.5 h per dataset on one NVIDIA A800.

