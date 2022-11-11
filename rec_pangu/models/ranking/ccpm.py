# -*- ecoding: utf-8 -*-
# @ModuleName: ccpm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from torch import nn
import torch
from ..layers import EmbeddingLayer, MLP_Layer, KMaxPooling, get_activation
from ..utils import get_feature_num, get_linear_input
from ..base_model import BaseModel
class CCPM(BaseModel):
    def __init__(self,
                 embedding_dim=32,
                 dnn_hidden_units=[64, 64, 64],
                 channels = [4, 4, 2],
                 kernel_heights=[6, 5, 3],
                 loss_fun='torch.nn.BCELoss()',
                 enc_dict=None):
        super(CCPM, self).__init__(enc_dict,embedding_dim)

        self.dnn_hidden_units = dnn_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.num_sparse, self.num_dense = get_feature_num(self.enc_dict)

        self.conv_layer = CCPM_ConvLayer(self.num_sparse,
                                         channels=channels,
                                         kernel_heights=kernel_heights)
        conv_out_dim = 3 * embedding_dim * channels[-1]  # 3 is k-max-pooling size of the last layer
        self.fc = nn.Linear(conv_out_dim, 1)

        self.apply(self._init_weights)

    def forward(self, data,is_training=True):

        feature_emb = self.embedding_layer(data)
        conv_in = torch.unsqueeze(feature_emb, 1)  # shape (bs, 1, field, emb)
        conv_out = self.conv_layer(conv_in)
        flatten_out = torch.flatten(conv_out, start_dim=1)
        y_pred = self.fc(flatten_out)

        y_pred = y_pred.sigmoid()
        # 输出
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1),data['label'])
            output_dict = {'pred':y_pred,'loss':loss}
        else:
            output_dict = {'pred':y_pred}
        return output_dict

class CCPM_ConvLayer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """
    def __init__(self, num_fields, channels=[3], kernel_heights=[3], activation="Tanh"):
        super(CCPM_ConvLayer, self).__init__()
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        elif len(kernel_heights) != len(channels):
            raise ValueError("channels={} and kernel_heights={} should have the same length."\
                             .format(channels, kernel_heights))
        module_list = []
        self.channels = [1] + channels
        layers = len(kernel_heights)
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, kernel_height - 1, kernel_height - 1)))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(KMaxPooling(k, dim=2))
            module_list.append(get_activation(activation))
        self.conv_layer = nn.Sequential(*module_list)

    def forward(self, X):
        return self.conv_layer(X)