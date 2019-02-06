import chainer
import teras.training as training


def chainer_train_on(*args, **kwargs):
    chainer.config.train = True
    chainer.config.enable_backprop = True


def chainer_train_off(*args, **kwargs):
    chainer.config.train = False
    chainer.config.enable_backprop = False


def set_random_seed(seed, device_id=-1):
    import os
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['CHAINER_SEED'] = os.environ['CUPY_SEED'] = str(seed)
    chainer.global_config.cudnn_deterministic = True
    if device_id >= 0:
        try:
            import cupy
            cupy.cuda.runtime.setDevice(device_id)
            cupy.random.seed(seed)
        except Exception as e:
            import teras.logging as logging
            logging.error(str(e))


def set_debug(debug):
    if debug:
        chainer.config.debug = True
        chainer.config.type_check = True
    else:
        chainer.config.debug = False
        chainer.config.type_check = False


class ExponentialDecayAnnealing(object):
    name = 'ExponentialDecayAnnealing'
    call_for_each_param = False

    def __init__(self, initial_lr, decay_rate, decay_step,
                 staircase=False, lr_key='lr'):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.staircase = staircase
        self.lr_key = lr_key
        self.step = 0

    def __call__(self, optimizer):
        self.step += 1
        p = self.step / self.decay_step
        if self.staircase:
            p //= 1
        lr = self.initial_lr * self.decay_rate ** p
        setattr(optimizer, self.lr_key, lr)


class Saver(training.listeners.Saver):
    name = "chainer.saver"

    def save_model(self, model, suffix=''):
        file = "{}{}.npz".format(self._basename, suffix)
        self._logger.info("saving the model to {} ...".format(file))
        chainer.serializers.save_npz(file, model)


set_debug(chainer.config.debug)
chainer.config.use_cudnn = 'auto'

training_config = {'hooks': {
    training.EPOCH_TRAIN_BEGIN: chainer_train_on,
    training.EPOCH_VALIDATE_BEGIN: chainer_train_off,
}}
