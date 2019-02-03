from teras.app import App, arg
from teras.io import cache
import teras.utils.logging as Log
from teras.utils import git

import dataset


def train(train_file, test_file=None, cache_dir='', refresh_cache=False):

    def _create():
        loader = dataset.DataLoader(input_file=train_file)
        train_dataset = loader.load(train_file, train=True)
        test_dataset = loader.load(test_file, train=False) \
            if test_file is not None else None
        return loader, train_dataset, test_dataset

    loader, train_dataset, test_dataset = \
        cache.load_or_create(key=(git.hash(), train_file, test_file),
                             factory=_create, refresh=refresh_cache,
                             dir=cache_dir, mkdir=True, logger=Log.getLogger())


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
