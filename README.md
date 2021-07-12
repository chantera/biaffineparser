# biaffineparser: Deep Biaffine Attention Dependency Parser

biaffineparser is a PyTorch implementation of "[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)."

## Installation

biaffineparser works on PyTorch.

```sh
$ git clone https://github.com/chantera/biaffineparser
$ cd biaffineparser
$ pip install -r requirements.txt
```

## Usage

### Training

```sh
usage: main.py train [-h] --train_file FILE [--eval_file FILE]
                     [--embed_file FILE] [--max_steps NUM]
                     [--eval_interval NUM] [--batch_size NUM]
                     [--learning_rate VALUE] [--cuda] [--save_dir DIR]
                     [--cache_dir DIR] [--seed VALUE]
```

### Evaluation

```sh
usage: main.py evaluate [-h] --eval_file FILE --checkpoint_file FILE
                        --preprocessor_file FILE [--output_file FILE]
                        [--batch_size NUM] [--cuda] [--verbose]
```

## Example

```sh
$ mkdir models
$ python3 src/main.py train --train_file $DATA/train.conll --eval_file $DATA/dev.conll --embed_file $DATA/glove.6B.100d.txt --cuda --save_dir ./models
$ python3 src/main.py evaluate --eval_file $DATA/test.conll --ckpt ./models/step-[num].ckpt --proc ./models/preprocessor.pt --cuda
```

### Performance

The model achieves **UAS: 95.77** and **LAS: 94.10** in wsj 23 (test set) in PTB-SD 3.3.0 with the reported hyperparameters.

## References

  - Dozat, T., Manning, C. D., 2016. Deep Biaffine Attention for Neural Dependency Parsing. arXiv preprint arXiv:1611.01734. <https://arxiv.org/abs/1611.01734>

License
----
Apache License Version 2.0

&copy; Copyright 2021 Hiroki Teranishi

