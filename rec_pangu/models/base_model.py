# -*- ecoding: utf-8 -*-
# @ModuleName: base_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/8/16 5:10 PM
import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
import numpy as np
from .layers import EmbeddingLayer
from loguru import logger


class BaseModel(nn.Module):
    def __init__(self, enc_dict: dict, embedding_dim: int) -> None:
        """
        A base class for a neural network model.

        Args:
            enc_dict (dict): A dictionary containing the encoding details.
            embedding_dim (int): Dimension of the embedding layer.
        """
        super().__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the model.

        Args:
            module (nn.Module): A neural network module.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def set_pretrained_weights(self, col_name: str, pretrained_dict: dict, trainable: bool = True) -> None:
        """
        Set the pre-trained weights for the model.

        Args:
            col_name (str): Column name for the embedding layer.
            pretrained_dict (dict): A pre-trained embedding dictionary.
            trainable (bool, optional): Training flag. Defaults to True.

        Raises:
            AssertionError: If the column name is not in the encoding dictionary.
                              If the pre-trained embedding dimension is not equal to the model embedding dimension.
        """
        assert col_name in self.enc_dict.keys(), "Pretrained Embedding Col: {} must be in the {}".format(col_name,
                                                                                                         self.enc_dict.keys())
        pretrained_emb_dim = len(list(pretrained_dict.values())[0])
        assert self.embedding_dim == pretrained_emb_dim, "Pretrained Embedding Dim:{} must be equal to Model Embedding Dim:{}".format(
            pretrained_emb_dim, self.embedding_dim)
        pretrained_emb = np.random.rand(self.enc_dict[col_name]['vocab_size'], pretrained_emb_dim)
        for k, v in self.enc_dict[col_name].items():
            if k == 'vocab_size':
                continue
            pretrained_emb[v, :] = pretrained_dict.get(k, np.random.rand(pretrained_emb_dim))

        embeddings = torch.from_numpy(pretrained_emb).float()
        embedding_matrix = torch.nn.Parameter(embeddings)
        self.embedding_layer.set_weights(col_name=col_name, embedding_matrix=embedding_matrix, trainable=trainable)
        logger.info(
            'Successfully Set The Pretrained Embedding Weights for the column:{} With Trainable={}'.format(col_name,
                                                                                                           trainable))


class SequenceBaseModel(nn.Module):
    """
    Base sequence model for recommendation tasks.

    Attributes:
    enc_dict (dict): A dictionary mapping categorical variable names to their respective encoding dictionaries.
    config (dict): A dictionary containing model hyperparameters such as the embedding size, max sequence length, and device.
    embedding_dim (int): The embedding dimension size.
    max_length (int): The maximum length for input sequences.
    device (str): The device on which the model is run.
    item_emb (nn.Embedding): An embedding layer for item features.
    loss_fun: (nn.CrossEntropyLoss): A loss function used for training.
    """

    def __init__(self, enc_dict: dict, config: dict):
        super().__init__()

        self.enc_dict = enc_dict
        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.device = self.config['device']

        self.item_emb = nn.Embedding(self.enc_dict[self.config['item_col']]['vocab_size'], self.embedding_dim,
                                     padding_idx=0)
        for col in self.config['cate_cols']:
            setattr(self, f'{col}_emb',
                    nn.Embedding(self.enc_dict[col]['vocab_size'], self.embedding_dim, padding_idx=0))

        self.loss_fun = nn.CrossEntropyLoss()

    def calculate_loss(self, user_emb: torch.Tensor, pos_item: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the model given a user embedding and positive item.

        Args:
        user_emb (torch.Tensor): A tensor representing the user embedding.
        pos_item (torch.Tensor): A tensor representing the positive item.

        Returns:
        The tensor representing the calculated loss value.
        """
        all_items = self.output_items()
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        loss = self.loss_fun(scores, pos_item)
        return loss

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        """
        Gathers the vectors at the specific positions over a minibatch.

        Args:
        output (torch.Tensor): A tensor representing the output vectors.
        gather_index (torch.Tensor): A tensor representing the index vectors.

        Returns:
        The tensor representing the gathered output vectors.
        """
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def output_items(self) -> torch.Tensor:
        """
        Returns the item embedding layer weight.

        Returns:
        The tensor representing the item embedding layer weight.
        """
        return self.item_emb.weight

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate left-to-right uni-directional attention mask for multi-head attention.

        Args:
        attention_mask: a tensor used in multi-head attention with shape (batch_size,
        seq_len), containing values of either 0 or 1. 0 indicates padding of a sequence
        and 1 indicates the actual content of the sequence.

        Return:
        extended_attention_mask: a tensor with shape (batch_size, 1, seq_len, seq_len).
        An attention mask tensor with float values of -1e6 added to masked positions
        and 0 to unmasked positions.
        """
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # reshape to (batch_size, 1, 1, seq_len)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8),
                                     diagonal=1)  # create a matrix of upper triangle

        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).type_as(
            attention_mask)  # reshape and convert to attention_mask type

        extended_attention_mask = extended_attention_mask * subsequent_mask  # apply mask

        extended_attention_mask = (
                                              1.0 - extended_attention_mask) * -1e6  # replace masked positions with -1e6 and unmasked positions with 0

        return extended_attention_mask

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weight value for the given module.

        Args:
        module (nn.Module): The module whose weights need to be initialized.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)


class GraphBaseModel(nn.Module):
    def __int__(self, num_user, num_item, embedding_dim):
        super(GraphBaseModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_user = num_item
        self.num_item = num_item

        self.user_emb_layer = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.num_item, self.embedding_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss

    def get_ego_embedding(self):
        user_emb = self.user_emb_layer.weight
        item_emb = self.item_emb_layer.weight

        return torch.cat([user_emb, item_emb], 0)
