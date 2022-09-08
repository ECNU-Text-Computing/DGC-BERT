# DGCBERT
Re-examining lexical and semantic attention: Dual-view graph convolutions enhanced BERT for academic paper rating

## Requirement

* pytorch >= 1.9.0
* numpy >= 1.13.3
* sklearn
* python 3.9
* transformers

## Usage
Download the dataset from XXX <br>
make directory ./checkpoints/$data_source/

### Training
```sh
# SciBERT
python main.py --phase SciBERT --data_source AAPR/PeerRead --type SciBERT
# DGCBERT
python main.py --phase DGCBERT --data_source AAPR/PeerRead --type SciBERT --mode top_biaffine+softmax --k X --alpha X --top_rate X --predict_dim X
```

### Testing

```sh
# SciBERT
python main.py --phase model_test --data_source AAPR/PeerRead --model SciBERT
# DGCBERT
python main.py --phase model_test --data_source AAPR/PeerRead --model DGCBERT
```