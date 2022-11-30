# DGCBERT
Re-examining lexical and semantic attention: Dual-view graph convolutions enhanced BERT for academic paper rating

## Requirement

* pytorch >= 1.9.0
* numpy >= 1.13.3
* sklearn
* python 3.9
* transformers

## Usage
Original dataset:
* AAPR: https://github.com/lancopku/AAPR
* PeerRead: https://github.com/allenai/PeerRead

Download the dealt dataset from https://drive.google.com/file/d/1UWQzGYuxL53PjNY6wmcknpwhllEEWCUl/view?usp=sharing <br>

You can get the SciBERT model here https://github.com/allenai/scibert <br>
And put the SciBERT model in ./bert/scibert/ <br>
You can get the pre-trained state_dicts here https://drive.google.com/file/d/13Inl_ChtY0LBCp9D0wjFW1yFdJPvRMep/view?usp=sharing

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