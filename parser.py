#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import chainer
from teras.app import App, arg
from teras.framework.chainer import (
    chainer_train_off,
    config as chainer_config,
    set_debug as chainer_debug)
import teras.logging as Log
from teras.training import Trainer, TrainEvent as Event
import teras.utils

from model import DeepBiaffine, BiaffineParser, DataLoader, Evaluator


def train(
        train_file,
        test_file,
        embed_file,
        embed_size=100,
        n_epoch=20,
        batch_size=32,
        lr=0.002,
        model_params={},
        gpu=-1,
        save_to=None):
    context = locals()

    # Load files
    Log.i('initialize DataLoader with embed_file={} and embed_size={}'
          .format(embed_file, embed_size))
    loader = DataLoader(word_embed_file=embed_file,
                        pos_embed_size=embed_size)
    Log.i('load train dataset from {}'.format(train_file))
    train_dataset = loader.load(train_file, train=True)
    Log.i('load test dataset from {}'.format(test_file))
    test_dataset = loader.load(test_file, train=False)

    model_cls = DeepBiaffine

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(model_cls))
    Log.i('# model params: {}'.format(model_params))
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network model
    model = model_cls(
        embeddings=(loader.get_embeddings('word'),
                    loader.get_embeddings('pos')),
        n_labels=len(loader.label_map),
        **model_params,
    )
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    chainer_debug(App.debug)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.9, eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    Log.i('optimizer: Adam(alpha={}, beta1=0.9, '
          'beta2=0.9, eps=1e-08), grad_clip=5.0'.format(lr))

    def annealing(data):
        decay, decay_step = 0.75, 5000
        optimizer.alpha = optimizer.alpha * \
            (decay ** (data['epoch'] / decay_step))

    # Setup a trainer
    parser = BiaffineParser(model)

    trainer = Trainer(optimizer, parser, loss_func=parser.compute_loss,
                      accuracy_func=parser.compute_accuracy)
    trainer.configure(chainer_config)
    trainer.add_hook(Event.EPOCH_END, annealing)
    trainer.attach_callback(
        Evaluator(parser, pos_map=loader.get_processor('pos').vocabulary,
                  ignore_punct=True))

    if save_to is not None:
        accessid = Log.getLogger().accessid
        date = Log.getLogger().accesstime.strftime('%Y%m%d')

        def _save(data):
            epoch = data['epoch']
            model_file = os.path.join(save_to, "{}-{}.{}.npz"
                                      .format(date, accessid, epoch))
            Log.i("saving the model to {} ...".format(model_file))
            chainer.serializers.save_npz(model_file, model)
        context['model_cls'] = model_cls
        context['loader'] = loader
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


def test(
        model_file,
        target_file,
        printout=False,
        gpu=-1):

    # Load context
    context = teras.utils.load_context(model_file)

    # Load files
    Log.i('load dataset from {}'.format(target_file))
    dataset = context.loader.load(target_file, train=False)

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(context.model_cls))
    Log.i('# context: {}'.format(context))
    Log.v('--------------------------------')
    Log.v('')

    # Set up a neural network model
    model = context.model_cls(
        embeddings=(context.loader.get_embeddings('word'),
                    context.loader.get_embeddings('pos')),
        n_labels=len(context.loader.label_map),
        **context.model_params,
    )
    chainer.serializers.load_npz(model_file, model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    chainer_debug(App.debug)

    parser = BiaffineParser(model)
    evaluator = \
        Evaluator(parser,
                  pos_map=context.loader.get_processor('pos').vocabulary,
                  ignore_punct=True)

    # Start testing
    chainer_train_off()
    UAS, LAS, count = 0, 0, 0
    for batch_index, batch in enumerate(
            dataset.batch(context.batch_size, shuffle=False)):
        word_tokens, pos_tokens = batch[:-1]
        true_arcs, true_labels = batch[-1].T
        arcs_batch, labels_batch = parser.parse(word_tokens, pos_tokens)
        for i, (p_arcs, p_labels, t_arcs, t_labels) in enumerate(
                zip(arcs_batch, labels_batch, true_arcs, true_labels)):
            mask = evaluator.create_ignore_mask(word_tokens[i], pos_tokens[i])
            _uas, _las, _count = evaluator.evaluate(
                p_arcs, p_labels, t_arcs, t_labels, mask)
            UAS, LAS, count = UAS + _uas, LAS + _las, count + _count
    Log.i("[evaluation] UAS: {:.8f}, LAS: {:.8f}"
          .format(UAS / count * 100, LAS / count * 100))


if __name__ == "__main__":
    corpus = '/Users/hiroki/Desktop/NLP/data/ptb-sd3.3.0/dep/'
    # datadir = App.basedir + '/../data/'
    datadir = '/Users/hiroki/Desktop/coord/data/'
    _default_train_file = corpus + 'wsj_02-21.conll'
    _default_valid_file = corpus + 'wsj_22.conll'
    _default_embed_file = datadir + 'ptb.200.vec'

    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', '-b', type=int, default=32,
            help='Number of examples in each mini-batch'),
        'embed_file':
        arg('--embedfile', type=str, default=_default_embed_file,
            help='Pretrained word embedding file'),
        'embed_size':
        arg('--embedsize', type=int, default=100,
            help='Size of embeddings'),
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'lr':
        arg('--lr', type=float, default=0.002,
            help='Learning Rate'),
        'model_params':
        arg('--model', action='store_dict', default={},
            help='Model hyperparameter'),
        'n_epoch':
        arg('--epoch', '-e', type=int, default=20,
            help='Number of sweeps over the dataset to train'),
        'save_to':
        arg('--out', type=str, default=None,
            help='Save model to the specified directory'),
        'test_file':
        arg('--validfile', type=str, default=_default_valid_file,
            help='validation data file'),
        'train_file':
        arg('--trainfile', type=str, default=_default_train_file,
            help='training data file'),
    })

    App.add_command('test', test, {
        'gpu':
        arg('--gpu', '-g', type=int, default=-1,
            help='GPU ID (negative value indicates CPU)'),
        'model_file':
        arg('--modelfile', type=str, required=True,
            help='Trained model archive file'),
        'printout':
        arg('--print', action='store_true', default=False,
            help='Print decoded coordination'),
        'target_file':
        arg('--targetfile', type=str, required=True,
            help='Decoding target data file'),
    })

    App.run()
