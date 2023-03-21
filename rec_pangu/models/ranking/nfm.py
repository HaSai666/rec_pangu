# -*- ecoding: utf-8 -*-
# @ModuleName: nfm
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from typing import Dict, List
from ..layers import LR_Layer, MLP, InnerProductLayer
from ..utils import get_dnn_input_dim
from ..base_model import BaseModel


class NFM(BaseModel):
    def __init__(self,
                 embedding_dim: int = 32,
                 hidden_units: List[int] = [64, 64, 64],
                 loss_fun: str = 'torch.nn.BCELoss()',
                 enc_dict: Dict[str, dict] = None):
        """
        Neural Factorization Machine (NFM) model.

        Args:
            embedding_dim (int, optional): The dimension of the embedding layer. Defaults to 32.
            hidden_units (List[int], optional): The number of hidden units in the DNN layers. Defaults to [64, 64, 64].
            loss_fun (str, optional): The loss function. Defaults to 'torch.nn.BCELoss()'.
            enc_dict (Optional[Dict[str, int]], optional): The encoding dictionary. Defaults to None.
        """
        super(NFM, self).__init__(enc_dict, embedding_dim)

        self.hidden_units = hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict

        self.lr = LR_Layer(enc_dict=self.enc_dict)

        self.inner_product_layer = InnerProductLayer(output="Bi_interaction_pooling")
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        self.dnn = MLP(input_dim=self.embedding_dim, output_dim=1, hidden_units=self.hidden_units,
                       hidden_activations='relu', dropout_rates=0)
        self.apply(self._init_weights)

    def forward(self, data: Dict[str, torch.Tensor], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform forward propagation on the NFM model.

        Args:
            data (Dict[str, torch.Tensor]): The input data in the form of a dictionary containing the features and labels.
            is_training (bool): If True, compute the loss. Default is True.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model predictions and loss (if is_training is True).
        """
        # Linear part
        y_pred = self.lr(data)  # Batch, 1
        batch_size = y_pred.shape[0]

        # Embedding part
        sparse_embedding = self.embedding_layer(data)

        # Bi-interaction pooling
        inner_product_tensor = self.inner_product_layer(sparse_embedding)
        bi_pooling_tensor = inner_product_tensor.view(batch_size, -1)

        # DNN part
        y_pred += self.dnn(bi_pooling_tensor)
        y_pred = y_pred.sigmoid()

        # Output
        if is_training:
            loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else:
            output_dict = {'pred': y_pred}

        return output_dict
