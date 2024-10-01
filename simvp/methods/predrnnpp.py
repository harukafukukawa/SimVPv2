import torch.nn as nn

from simvp.models import PredRNNpp_Model
from .predrnn import PredRNN
from .crack_area_mse import CrackAreaMSE


class PredRNNpp(PredRNN):
    r"""PredRNN++

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
#        self.criterion = nn.MSELoss()
        self.criterion = CrackAreaMSE()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNpp_Model(num_layers, num_hidden, args).to(self.device)
