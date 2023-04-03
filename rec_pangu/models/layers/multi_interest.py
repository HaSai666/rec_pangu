# -*- ecoding: utf-8 -*-
# @ModuleName: multi_interest_layer
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 15:12
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInterestSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_attention_heads: int, d: int = None) -> None:
        super(MultiInterestSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        if d is None:
            self.d = self.embedding_dim * 4
        else:
            self.d = d

        # Create trainable parameters
        self.W1 = nn.Parameter(torch.rand(self.embedding_dim, self.d), requires_grad=True)
        self.W2 = nn.Parameter(torch.rand(self.d, self.num_attention_heads), requires_grad=True)
        self.W3 = nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, sequence_embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        '''
        Args:
            * sequence_embeddings (torch.Tensor): batch_size x sequence_length x embedding_dimension
            * mask (torch.Tensor): binary mask for sequence; batch_size x sequence_length x 1

        Returns:
            * Multi-interest embeddings (torch.Tensor): batch_size x num_attention_heads x embedding_dimension
        '''

        # Apply first layer of weights and activation function
        H = torch.einsum('bse, ed -> bsd', sequence_embeddings, self.W1).tanh()

        # Apply second layer of weights and softmax
        if mask is not None:
            # Add a large negative constant to the masked elements of the attention matrix to give them near 0 probability.
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)

        # Transpose attention head and sequence dimensions for easier matrix multiplication
        A = A.permute(0, 2, 1)

        # Apply final attention pooling to get multi-interest embeddings
        multi_interest_emb = torch.matmul(A, sequence_embeddings)

        return multi_interest_emb


class CapsuleNetwork(nn.Module):

    def __init__(self, hidden_size: int, seq_len: int, bilinear_type: int = 2, interest_num: int = 4,
                 routing_times: int = 3, hard_readout: bool = True, relu_layer: bool = False) -> None:
        """
        Implements a Capsule Network that is capable of handling various types of bilinear
        interactions between items in a sequence.

        Args:
        hidden_size: An integer representing the size of the hidden layer of the model.
        seq_len: An integer representing the length of the input sequence.
        bilinear_type: An integer representing the type of bilinear interaction between items.
        interest_num: An integer representing the number of interest capsules in the model.
        routing_times: An integer representing the number of dynamic routing iterations.
        hard_readout: A Boolean indicating whether to use hard readout or not
        relu_layer: A Boolean indicating whether to use a ReLU layer

        Returns:
        interest_capsule: The output interest capsule from the model
        """
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = False
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU()
        )
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias=False)
        else:  # ComiRec_DR
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size))

    def forward(self, item_eb, mask, device):
        if self.bilinear_type == 0:  # MIND
            item_eb_hat = self.linear(item_eb)  # [b, s, h]
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)  # [b, s, h*in]
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:  # ComiRec_DR
            u = torch.unsqueeze(item_eb, dim=2)  # shape=(batch_size, maxlen, 1, embedding_dim)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u,
                                    dim=3)  # shape=(batch_size, maxlen, hidden_size*interest_num)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        # [b, in, s, h]
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # b的shape=(b, in, s)
        if self.bilinear_type > 0:  # b初始化为0（一般的胶囊网络算法）
            capsule_weight = torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device,
                                         requires_grad=False)
        else:  # MIND使用高斯分布随机初始化b
            capsule_weight = torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=device,
                                         requires_grad=False)

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)  # [b, in, s]
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            # 计算c，进行mask，最后shape=[b, in, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)  # mask
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                # s=c*u_hat , (batch_size, interest_num, 1, seq_len) * (batch_size, interest_num, seq_len, hidden_size)
                interest_capsule = torch.matmul(capsule_softmax_weight,
                                                item_eb_hat_iter)  # shape=(batch_size, interest_num, 1, hidden_size)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)  # shape=(batch_size, interest_num, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)  # shape同上
                interest_capsule = scalar_factor * interest_capsule  # squash(s)->v,shape=(batch_size, interest_num, 1, hidden_size)

                # 更新b
                delta_weight = torch.matmul(item_eb_hat_iter,  # shape=(batch_size, interest_num, seq_len, hidden_size)
                                            torch.transpose(interest_capsule, 2, 3).contiguous()
                                            # shape=(batch_size, interest_num, hidden_size, 1)
                                            )  # u_hat*v, shape=(batch_size, interest_num, seq_len, 1)
                delta_weight = torch.reshape(delta_weight, (
                    -1, self.interest_num, self.seq_len))  # shape=(batch_size, interest_num, seq_len)
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        if self.relu_layer:  # MIND模型使用book数据库时，使用relu_layer
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule
