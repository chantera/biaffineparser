import chainer
import numpy as np
from teras.app import App, arg
import teras.training as training
from teras.utils import git, logging
import torch
from tqdm import tqdm

from common import optimizers, utils
import dataset
from eval import Evaluator
import models


chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)
logging.captureWarnings(True)


def train(train_file, test_file=None, embed_file=None,
          n_epoch=20, batch_size=5000, lr=2e-3, model_config=None, device=-1,
          save_dir=None, seed=None, cache_dir='', refresh_cache=False):
    if seed is not None:
        utils.set_random_seed(seed, device)
    logger = logging.getLogger()
    assert isinstance(logger, logging.AppLogger)
    if model_config is None:
        model_config = {}

    loader = dataset.DataLoader.build(
        input_file=train_file, word_embed_file=embed_file,
        refresh_cache=refresh_cache, extra_ids=(git.hash(),),
        cache_options=dict(dir=cache_dir, mkdir=True, logger=logger))
    train_dataset = loader.load(train_file, train=True, bucketing=True,
                                refresh_cache=refresh_cache)
    test_dataset = None
    if test_file is not None:
        test_dataset = loader.load(test_file, train=False, bucketing=True,
                                   refresh_cache=refresh_cache)

    model = _build_parser(loader, **model_config)
    if device >= 0:
        model.cuda()
    #     chainer.cuda.get_device_from_id(device).use()
    #     model.to_gpu(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr, betas=(0.9, 0.9), eps=1e-12)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    # optimizer.add_hook(optimizers.ExponentialDecayAnnealing(
    #     initial_lr=lr, decay_rate=0.75, decay_step=5000, lr_key='alpha'))

    def _report(y, t):
        arc_accuracy, rel_accuracy = model.compute_accuracy(y, t)
        training.report({'arc_accuracy': arc_accuracy,
                         'rel_accuracy': rel_accuracy})

    trainer = training.Trainer(optimizer, model, loss_func=model.compute_loss)
    trainer.configure(utils.training_config)
    trainer.add_listener(
        training.listeners.ProgressBar(lambda n: tqdm(total=n)), priority=200)
    trainer.add_hook(
        training.BATCH_END, lambda data: _report(data['ys'], data['ts']))
    if test_dataset:
        evaluator = Evaluator(model, loader.rel_map, test_file, logger)
        trainer.add_listener(evaluator, priority=128)
        if save_dir is not None:
            accessid = logger.accessid
            date = logger.accesstime.strftime('%Y%m%d')
            trainer.add_listener(
                utils.Saver(model, basename="{}-{}".format(date, accessid),
                            context=dict(App.context, loader=loader),
                            directory=save_dir, logger=logger, save_best=True,
                            evaluate=(lambda _: evaluator._parsed['UAS'])))
    trainer.fit(train_dataset, test_dataset, n_epoch, batch_size)


def test(model_file, test_file, device=-1):
    context = utils.Saver.load_context(model_file)
    if context.seed is not None:
        utils.set_random_seed(context.seed, device)

    test_dataset = context.loader.load(test_file, train=False, bucketing=True)
    kwargs = dict(context)
    if context.model_config is not None:
        kwargs.update(context.model_config)
    model = _build_parser(**dict(kwargs))
    chainer.serializers.load_npz(model_file, model)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)

    pbar = training.listeners.ProgressBar(lambda n: tqdm(total=n))
    pbar.init(len(test_dataset))
    evaluator = Evaluator(
        model, context.loader.rel_map, test_file, logging.getLogger())
    utils.chainer_train_off()
    for batch in test_dataset.batch(
            context.batch_size, colwise=True, shuffle=False):
        xs, ts = batch[:-1], batch[-1]
        ys = model.forward(*xs)
        evaluator.on_batch_end({'train': False, 'xs': xs, 'ys': ys, 'ts': ts})
        pbar.update(len(ts))
    evaluator.on_epoch_validate_end({})


def _build_parser(loader, **kwargs):
    dropout_ratio = kwargs.get('dropout', 0.33)
    parser = models.BiaffineParser(
        n_rels=len(loader.rel_map),
        encoder=models.Encoder(
            loader.get_embeddings('word'),
            loader.get_embeddings('pre', normalize=lambda W: W / np.std(W)
                                  if np.std(W) > 0. else W),
            loader.get_embeddings('pos'),
            n_lstm_layers=kwargs.get('n_lstm_layers', 3),
            lstm_hidden_size=kwargs.get('lstm_hidden_size', 400),
            embeddings_dropout=kwargs.get('input_dropout', dropout_ratio),
            lstm_dropout=kwargs.get('lstm_dropout', dropout_ratio),
            recurrent_dropout=kwargs.get('recurrent_dropout', dropout_ratio)),
        arc_mlp_units=kwargs.get('arc_mlp_units', 500),
        rel_mlp_units=kwargs.get('rel_mlp_units', 100),
        arc_mlp_dropout=kwargs.get('arc_mlp_dropout', dropout_ratio),
        rel_mlp_dropout=kwargs.get('rel_mlp_dropout', dropout_ratio))
    return parser


if __name__ == "__main__":
    App.configure(logdir=App.basedir + '/../logs', loglevel='debug')
    logging.AppLogger.configure(mkdir=True)
    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', type=int, default=5000, metavar='NUM',
            help='Number of tokens in each mini-batch'),
        'cache_dir':
        arg('--cachedir', type=str, default=(App.basedir + '/../cache'),
            metavar='DIR', help='Cache directory'),
        'test_file':
        arg('--devfile', type=str, default=None, metavar='FILE',
            help='Development data file'),
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'embed_file':
        arg('--embedfile', type=str, default=None, metavar='FILE',
            help='Pretrained word embedding file'),
        'n_epoch':
        arg('--epoch', type=int, default=300, metavar='NUM',
            help='Number of sweeps over the dataset to train'),
        'lr':
        arg('--lr', type=float, default=2e-3, metavar='VALUE',
            help='Learning rate'),
        'model_config':
        arg('--model', action='store_dict', metavar='KEY=VALUE',
            help='Model configuration'),
        'refresh_cache':
        arg('--refresh', '-r', action='store_true', help='Refresh cache.'),
        'save_dir':
        arg('--savedir', type=str, default=None, metavar='DIR',
            help='Directory to save the model'),
        'seed':
        arg('--seed', type=int, default=None, metavar='VALUE',
            help='Random seed'),
        'train_file':
        arg('--trainfile', type=str, required=True, metavar='FILE',
            help='Training data file.'),
    })
    App.add_command('test', test, {
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'model_file':
        arg('--modelfile', type=str, required=True, metavar='FILE',
            help='Trained model file'),
        'test_file':
        arg('--testfile', type=str, required=True, metavar='FILE',
            help='Test data file'),
    })
    App.run()
