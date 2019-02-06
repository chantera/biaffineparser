import chainer
from teras.app import App, arg
from teras.io import cache
import teras.training as training
import teras.utils.logging as Log
from teras.utils import git
from tqdm import tqdm

import dataset
import models
import utils


chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)


def train(train_file, test_file=None,
          epoch=20, batch_size=32, lr=0.001,
          device=-1, save_dir=None, seed=None,
          cache_dir='', refresh_cache=False):
    if seed is not None:
        utils.set_random_seed(seed, device)
    logger = Log.getLogger()
    assert isinstance(logger, Log.AppLogger)

    def _load():
        loader = dataset.DataLoader(input_file=train_file)
        train_dataset = loader.load(train_file, train=True)
        test_dataset = loader.load(test_file, train=False) \
            if test_file is not None else None
        return loader, train_dataset, test_dataset

    loader, train_dataset, test_dataset = \
        cache.load_or_create(key=(git.hash(), train_file, test_file),
                             factory=_load, refresh=refresh_cache,
                             dir=cache_dir, mkdir=True, logger=logger)

    model = models.BiaffineParser(
        n_rels=len(loader.rel_map),
        encoder=models.Encoder(
            loader.get_embeddings('word'),
            loader.get_embeddings('pre'),
            loader.get_embeddings('pos'),
            n_lstm_layers=3,
            lstm_hidden_size=400,
            embeddings_dropout=0.33,
            lstm_dropout=0.33),
        encoder_dropout=0.33,
        arc_mlp_units=500, rel_mlp_units=100,
        arc_mlp_dropout=0.33, rel_mlp_dropout=0.33)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)
    optimizer = chainer.optimizers.Adam(
        alpha=lr, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    # optimizer.add_hook(utils.ExponentialDecayAnnealing(
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
        trainer.add_listener(
            utils.Evaluator(model, loader.rel_map, test_file, logger))
    if save_dir is not None:
        accessid = logger.accessid
        date = logger.accesstime.strftime('%Y%m%d')
        trainer.add_listener(
            utils.Saver(model, basename="{}-{}".format(date, accessid),
                        context=dict(App.context, loader=loader),
                        directory=save_dir, logger=logger))
    trainer.fit(train_dataset, test_dataset, epoch, batch_size)


if __name__ == "__main__":
    App.configure(logdir=App.basedir + '/../logs')
    Log.AppLogger.configure(mkdir=True)
    App.add_command('train', train, {
        'train_file':
        arg('--trainfile', type=str, required=True,
            help='Training data file.'),
        'test_file':
        arg('--devfile', type=str, default=None,
            help='Development data file'),
        'save_dir':
        arg('--savedir', type=str, default=None,
            help='Directory to save the model'),
        'cache_dir':
        arg('--cachedir', type=str, default=(App.basedir + '/../cache'),
            help='Cache directory'),
        'refresh_cache':
        arg('--refresh', action='store_true', help='Refresh cache.'),
    })
    App.run()
