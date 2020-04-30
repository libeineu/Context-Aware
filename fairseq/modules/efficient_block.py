import torch
import torch.nn as nn
from fairseq.models.transformer import LayerNorm
import queue
import fairseq.utils as utils
import torch.nn.functional as F
import numpy as np


def CreateEfficientBlock(args, is_encoder):
    history_type = args.encoder_history_type if is_encoder else args.decoder_history_type
    if history_type is None:
        return None
    elif history_type == "residual":
        return ResidualLayerHistory(args, is_encoder)
    elif history_type == "dense":
        return DenseLayerHistory(args, is_encoder)
    elif history_type == "learnable_dense":
        return LearnableDenseLayerHistory(args, is_encoder)
    elif history_type == "learnable_dense_mask":
        return LearnableDenseMaskLayerHistory(args, is_encoder)
    elif history_type == "learnable_dense_nonorm":
        return LearnableDenseNoNormLayerHistory(args, is_encoder)
    else:
        raise ValueError


class BaseLayerHistory(nn.Module):

    def __init__(self, args, is_encoder):
        super(BaseLayerHistory, self).__init__()
        self.is_encoder = is_encoder
        self.normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before

        # the first layer (aka. embedding layer) does not have layer normalization
        layers = args.in_block_num
        dim = args.encoder_embed_dim
        self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layers))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented

    def reset(self, x):
        raise NotImplemented


class ResidualLayerHistory(BaseLayerHistory):
    """
    x_n = x_{n-1} + y_{n-1}
    """

    def __init__(self, args, is_encoder):
        super(ResidualLayerHistory, self).__init__(args, is_encoder)
        self.count = 0
        self.x = None
        self.y = None

    def add(self, layer):
        if self.x is None:
            self.x = layer
            self.count += 1
            return
        self.count += 1
        if self.normalize_before:
            self.y = self.layer_norms[self.count - 2](layer)
        else:
            self.y = layer

    def pop(self):
        assert self.x is not None
        if self.y is None:
            return self.x
        ret = self.x + self.y
        if not self.normalize_before:
            ret = self.layer_norms[self.count - 2](ret)
        self.x = ret
        return ret

    def clean(self):
        self.x = None
        self.y = None
        self.count = 0


class DenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder):
        super(DenseLayerHistory, self).__init__(args, is_encoder)
        self.sum = None
        self.count = 0
        self.individuals = None  # store past individual value, used for windows_size > 0

        self.integration_type = getattr(args, 'encoder_integration_type', 'avg') if is_encoder else \
            getattr(args, 'decoder_integration_type', 'avg')
        # windows = 1 means not use residual connection
        self.windows_size = getattr(args, 'encoder_windows_size', -1) if is_encoder else \
            getattr(args, 'decoder_windows_size', -1)
        if self.windows_size > 0:
            assert self.windows_size <= (args.encoder_layers + 1) if is_encoder else (args.decoder_layers + 1)
            self.individuals = queue.Queue(self.windows_size)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            if self.individuals is not None:
                self.individuals.put(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.sum = self.sum + layer
        if self.windows_size != -1 and self.count > self.windows_size:
            self.sum = self.sum - self.individuals.get()

        if self.individuals is not None:
            self.individuals.put(layer)

    def pop(self):
        assert self.sum is not None
        if self.integration_type == 'sum':
            ret = self.sum
        else:
            if self.windows_size == -1:
                ret = self.sum / self.count
            else:
                ret = self.sum / min(self.count, self.windows_size)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        if self.individuals is not None:
            self.individuals.queue.clear()

    def reset(self, x):
        self.sum = x
        self.count = 1


class LearnableDenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder):
        super(LearnableDenseLayerHistory, self).__init__(args, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)

        # print('count:', len(list(self.named_parameters())))
        # for k,v in self.named_parameters():
        #    print('k=%s' %k)

    def extra_repr(self):
        return 'n_layers={layer_num}, '.format(**self.__dict__)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        # layers_dropout = F.dropout(torch.stack(self.layers, 0), p=self.dense_dropout, training=self.training)
        # ret = (layers_dropout * self.weight[self.count -1, : self.count].view(-1, 1, 1, 1)).sum(0)
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()


class LearnableDenseMaskLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder):
        super(LearnableDenseMaskLayerHistory, self).__init__(args, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)
        if is_encoder:
            self.weight_mask = np.loadtxt("encoder_mask.txt", dtype=float, delimiter=' ')
        else:
            self.weight_mask = np.loadtxt("decoder_mask.txt", dtype=float, delimiter=' ')
        # self.weight_mask = torch.from_numpy(self.weight_mask).float()
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        # self.weight.data = self.weight.data * self.weight_mask
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)
        # self.weight_mask = self.weight_mask.cuda().half()

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        # layers_dropout = F.dropout(torch.stack(self.layers, 0), p=self.dense_dropout, training=self.training)
        # ret = (layers_dropout * self.weight[self.count -1, : self.count].view(-1, 1, 1, 1)).sum(0)
        # ret = (torch.stack(self.layers, 0) * (self.weight * self.weight_mask)[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()


class LearnableDenseNoNormLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder):
        super(LearnableDenseNoNormLayerHistory, self).__init__(args, is_encoder)
        self.sum = None
        self.count = 0
        self.layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)
        self.layers = []
        self.layer_norms = None

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0

        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []
