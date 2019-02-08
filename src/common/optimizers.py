class ExponentialDecayAnnealing(object):
    name = 'ExponentialDecayAnnealing'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, initial_lr, decay_rate, decay_step,
                 staircase=False, lr_key='lr'):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.staircase = staircase
        self.lr_key = lr_key
        self.step = 0

    def __call__(self, optimizer):
        p = self.step / self.decay_step
        if self.staircase:
            p //= 1
        lr = self.initial_lr * self.decay_rate ** p
        setattr(optimizer, self.lr_key, lr)
        self.step += 1
