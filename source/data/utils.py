from torch import nn


class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadows = {}

    def __len__(self):
        return len(self.shadows)

    def get(self, name: str):
        return self.shadows[name]

    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data

    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name).to(data.device)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow
