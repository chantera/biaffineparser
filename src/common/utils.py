import torch
import teras.training as training


def set_random_seed(seed):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Saver(training.listeners.Saver):
    name = "torch.saver"

    def save_model(self, model, suffix=''):
        file = "{}{}.pth".format(self._basename, suffix)
        self._logger.info("saving the model to {} ...".format(file))
        torch.save(model.state_dict(), file)
