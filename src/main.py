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
          epoch=20, batch_size=32,
          cache_dir='', refresh_cache=False):

    def _load():
        loader = dataset.DataLoader(input_file=train_file)
        train_dataset = loader.load(train_file, train=True)
        test_dataset = loader.load(test_file, train=False) \
            if test_file is not None else None
        return loader, train_dataset, test_dataset

    loader, train_dataset, test_dataset = \
        cache.load_or_create(key=(git.hash(), train_file, test_file),
                             factory=_load, refresh=refresh_cache,
                             dir=cache_dir, mkdir=True, logger=Log.getLogger())

    epoch = 1
    model = models.BiaffineParser(
        n_rels=len(loader.rel_map),
        encoder=models.Encoder(
            loader.get_embeddings('word'),
            loader.get_embeddings('pre'),
            loader.get_embeddings('pos'),
            n_lstm_layers=3,
            lstm_hidden_size=None,
            embeddings_dropout=0.33,
            lstm_dropout=0.0),
        arc_mlp_units=500, rel_mlp_units=100,
        arc_mlp_dropout=0.33, rel_mlp_dropout=0.33)
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    trainer = training.Trainer(optimizer, model, loss_func=model.compute_loss)
    trainer.configure(utils.training_config)
    trainer.add_listener(
        training.listeners.ProgressBar(lambda n: tqdm(total=n)), priority=200)
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
        'cache_dir':
        arg('--cachedir', type=str, default=(App.basedir + '/../cache'),
            help='Cache directory.', ),
        'refresh_cache':
        arg('--refresh', action='store_true', help='Refresh cache.'),
    })
    App.run()
