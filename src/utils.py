import chainer
from teras.training.event import TrainEvent


def chainer_train_on(*args, **kwargs):
    chainer.config.train = True
    chainer.config.enable_backprop = True


def chainer_train_off(*args, **kwargs):
    chainer.config.train = False
    chainer.config.enable_backprop = False


def set_random_seed(seed, gpu=-1):
    import os
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['CHAINER_SEED'] = os.environ['CUPY_SEED'] = seed
    chainer.global_config.cudnn_deterministic = True
    if gpu >= 0:
        try:
            import cupy
            cupy.cuda.runtime.setDevice(gpu)
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


def set_model_to_device(model, device_id=-1):
    if device_id >= 0:
        chainer.cuda.get_device_from_id(device_id).use()
        model.to_gpu()
    else:
        model.to_cpu()


set_debug(chainer.config.debug)
chainer.config.use_cudnn = 'auto'

training_config = {'hooks': {
    TrainEvent.EPOCH_TRAIN_BEGIN: chainer_train_on,
    TrainEvent.EPOCH_VALIDATE_BEGIN: chainer_train_off,
}}
