#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import chainer
from teras.app import App, arg
from teras.framework.chainer import (
    config as chainer_config,
    set_debug as chainer_debug)
import teras.logging as Log
from teras.training import Trainer, TrainEvent as Event
import teras.utils

from model import BaselineParser, DatasetProcessor, compute_accuracy, compute_cross_entropy
from utils import DEVELOP


def train(
        train_file,
        test_file,
        embed_file=None,
        n_epoch=20,
        batch_size=20,
        lr=0.001,
        reg_lambda=0.0001,
        loss_func='cross_entropy',
        embed_size=50,
        model_arch=2,
        lstm_hidden_size=600,
        use_gru=False,
        dropout_ratio=0.50,
        gpu=-1,
        save_to=None):
    context = locals()

    # Load files
    Log.i('initialize DatasetProcessor with embed_file={} and embed_size={}'
          .format(embed_file, embed_size))
    processor = DatasetProcessor(word_embed_file=embed_file,
                                 pos_embed_size=embed_size)
    Log.i('load train dataset from {}'.format(train_file))
    train_dataset = processor.load(train_file, train=True)
    Log.i('load test dataset from {}'.format(test_file))
    test_dataset = processor.load(test_file, train=False)

    cls = BaselineParser

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(cls))
    Log.i('# embed size: {}'.format(embed_size))
    Log.i('# n blstm layers: {}'.format(3))
    Log.i('# lstm hidden size: {}'.format(lstm_hidden_size))
    Log.i('# n mlp layers: {}'.format(2))
    Log.i('# dropout ratio: {:.4f}'.format(dropout_ratio))
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network model
    model = cls(
        embeddings=(processor.word_embeddings, processor.pos_embeddings),
        n_labels=39,
        # char_embeddings=(processor.char_embeddings
        #                  if model_arch == 7 else None),
        n_blstm_layers=3,
        lstm_hidden_size=lstm_hidden_size,
        use_gru=use_gru,
        # n_mlp_layers=2,
        dropout=dropout_ratio,
    )
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    chainer_debug(App.debug or DEVELOP)

    # loss_func = select_loss_func(loss_func)
    Log.i('loss function: {}'.format(loss_func))

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(reg_lambda))
    Log.i('optimizer: Adam(alpha={}, beta1=0.9, '
          'beta2=0.999, eps=1e-08), regularization: WeightDecay(lambda={})'
          .format(lr, reg_lambda))

    # Setup a trainer
    accessid = Log.getLogger().accessid
    date = Log.getLogger().accesstime.strftime('%Y%m%d')

    trainer = Trainer(optimizer, model, loss_func=compute_cross_entropy,
                      accuracy_func=compute_accuracy)
    trainer.configure(chainer_config)
    # trainer.attach_callback(Evaluator())

    if save_to is not None:
        def _save(data):
            epoch = data['epoch']
            model_file = os.path.join(save_to, "{}-{}.{}.npz"
                                      .format(date, accessid, epoch))
            Log.i("saving the model to {} ...".format(model_file))
            chainer.serializers.save_npz(model_file, model)
        context['n_blstm_layers'] = 3
        context['n_mlp_layers'] = 2
        context['word_vocab_size'] = processor.word_embeddings.shape[0]
        context['word_embed_size'] = processor.word_embeddings.shape[1]
        context['pos_vocab_size'] = processor.pos_embeddings.shape[0]
        context['char_vocab_size'] = processor.char_embeddings.shape[0]
        context['preprocessor'] = processor
        context_file = os.path.join(save_to, "{}-{}.context"
                                    .format(date, accessid))
        with open(context_file, 'wb') as f:
            teras.utils.dump(context, f)
        trainer.add_hook(Event.EPOCH_END, _save)

    # Start training
    trainer.fit(train_dataset, None,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=test_dataset,
                verbose=App.verbose)


if __name__ == "__main__":
    corpus = '/Users/hiroki/Desktop/NLP/data/ptb-sd3.3.0/dep/'
    datadir = App.basedir + '/../data/'
    _default_train_file = corpus + \
        ('wsj_02-21.conll' if not DEVELOP else 'wsj_02.conll')
    _default_valid_file = corpus + 'wsj_22.conll'
    _default_embed_file = datadir + 'ptb.200.vec'
    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', '-b', type=int, default=20,
            help='Number of examples in each mini-batch'),
        'dropout_ratio':
        arg('--dropout', '-dr', type=float, default=0.50,
            help='dropout ratio', metavar='RATIO'),
        'embed_file':
        arg('--embedfile', type=str, default=_default_embed_file,
            help='Pretrained word embedding file'),
        'embed_size':
        arg('--embedsize', type=int, default=50,
            help='Size of embeddings'),
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'loss_func':
        arg('--lossfun', type=str, default='cross_entropy',
            choices=('cross_entropy', 'hinge', 'squared_hinge'),
            help='Loss function'),
        'lr':
        arg('--lr', type=float, default=0.001,
            help='Learning Rate'),
        'lstm_hidden_size':
        arg('--lstmsize', type=int, default=600,
            help='Size of LSTM hidden vector'),
        'model_arch':
        arg('--model', '-m', type=int, default=2,
            help='Model Architecture'),
        'n_epoch':
        arg('--epoch', '-e', type=int, default=20,
            help='Number of sweeps over the dataset to train'),
        'reg_lambda':
        arg('--lambda', type=float, default=0.0001,
            help='L2 regularization rate'),
        'save_to':
        arg('--out', type=str, default=None,
            help='Save model to the specified directory'),
        'test_file':
        arg('--validfile', type=str, default=_default_valid_file,
            help='validation data file'),
        'train_file':
        arg('--trainfile', type=str, default=_default_train_file,
            help='training data file'),
        'use_gru':
        arg('--gru', action='store_true', default=False,
            help='use GRU as LSTM'),
    })
    App.run()
