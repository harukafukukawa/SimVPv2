import torch.nn as nn

from simvp.models import ConvLSTM_Model
from .predrnn import PredRNN
from .crack_area_mse import CrackAreaMSE

class ConvLSTM(PredRNN):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device,  steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
#        self.criterion = nn.MSELoss()
        self.criterion = CrackAreaMSE()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ConvLSTM_Model(num_layers, num_hidden, args).to(self.device)
