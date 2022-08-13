from turtle import forward
from torch import nn
from ..geometry import orthogonal, perspective, index

class BaseWAIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal'
                 ):
        super(BaseWAIFuNet, self).__init__()

        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective
        self.index = index
    
    def forward(self):
        pass