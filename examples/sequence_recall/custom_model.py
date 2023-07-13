# -*- ecoding: utf-8 -*-
# @ModuleName: custom_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/6 13:46
import sys

from torch import nn

sys.path.append('../../')

import torch
import torch.nn.functional as F
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.models.layers import BERT4RecEncoder, GRU4RecEncoder, CaserEncoder
from rec_pangu.trainer import SequenceTrainer
from rec_pangu.utils import set_device
import pandas as pd

class CustomMOEModel(SequenceBaseModel):
    '''
    自定义模型，对于单纯的序列推荐/多兴趣推荐，只需要继承SequenceBaseModel写好init和forward方法即可，其他部分均可复用
    '''

    def __init__(self, enc_dict, config):
        super(CustomMOEModel, self).__init__(enc_dict, config)

        self.num_expert = self.config.get('num_expert', 8)
        self.num_interest = self.config.get('num_interest', 4)
        self.num_layers = self.config.get('num_layers', 2)
        self.num_head = self.config.get('num_head', 4)
        self.init_moe_param(self.num_expert, self.num_interest)
        self.encoder_name = self.config.get("encoder_name", "BERT4Rec")
        self.encoder = self.init_encoder(self.encoder_name)
        self.reset_parameters()
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def init_encoder(self, encoder_name):
        if encoder_name == 'GRU4Rec':
            encoder = GRU4RecEncoder(self.embedding_dim, hidden_size=128)
        elif encoder_name == 'Caser':
            encoder = CaserEncoder(self.embedding_dim, self.max_length, num_horizon=16, num_vertical=8, l=5)
        elif encoder_name == 'BERT4Rec':
            encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=self.num_layers,
                                      num_heads=self.num_head)
        else:
            raise ValueError('Invalid sequence encoder.')
        return encoder

    def get_aug_emb(self, item_emb):
        random_noise = torch.rand_like(item_emb, device=item_emb.device)
        item_emb = item_emb + torch.sign(item_emb) * F.normalize(random_noise, dim=-1) * self.config.get('eps', 1e-3)
        return item_emb

    def cauclate_infonce_loss(self, item):
        '''
        自定义infonce loss
        '''
        item_emb = self.output_items()
        item_emb1 = self.get_aug_emb(item_emb)
        item_emb2 = self.get_aug_emb(item_emb)

        pos_item1 = item_emb1[item]
        pos_item2 = item_emb2[item]

        pos_item1 = F.normalize(pos_item1, dim=-1)
        pos_item2 = F.normalize(pos_item2, dim=-1)
        item_emb2 = F.normalize(item_emb2, dim=-1)

        pos_score = (pos_item2 * pos_item1).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.config.get('temperature', 0.1))
        ttl_score = torch.matmul(pos_item1, item_emb2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.config.get('temperature', 0.1)).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum() * self.config.get('alpha', 1e-7)

    def cauclate_uniformity_loss(self, multi_interest_emb):
        '''
        自定义uniformity loss
        '''
        multi_interest_emb = self.projection(multi_interest_emb)
        batch_size, view_number, _ = multi_interest_emb.shape
        all_difference = []
        for i in range(view_number):
            indices = list(range(view_number))
            indices.remove(i)
            other_views = torch.index_select(multi_interest_emb, 1, torch.tensor(indices).to(self.device))
            # have checked [batch_size, view_number-1, embedding_size]
            indices = [i]
            now_view = torch.index_select(multi_interest_emb, 1, torch.tensor(indices).to(self.device))
            # similarity = torch.mean(
            #     torch.sum(torch.abs(torch.bmm(other_views, now_view.transpose(1, -1))).squeeze(-1), dim=-1))
            difference = other_views - now_view
            difference = torch.sqrt(torch.sum(difference ** 2, dim=-1))
            difference = torch.mean(torch.mean(torch.exp(-difference), dim=-1), dim=-1)
            all_difference.append(difference.unsqueeze(-1))
        loss = torch.mean(torch.cat(all_difference))
        return loss

    def init_moe_param(self, num_expert, num_interest):
        self.gate_linear_list = nn.ModuleList(
            [nn.Linear(self.embedding_dim, num_expert) for _ in range(num_interest)])

    def moe(self, seq_emb, encoder_list, key_padding_mask):
        '''
        完成多个expert生产多个seq emb，然后对每个interest都对seq emb进行加权求和，完成多个interest的生产
        '''
        # 计算每一个expert的emb
        expert_emb_list = []
        for encoder in encoder_list:
            expert_emb_list.append(encoder(seq_emb, key_padding_mask))

        # 计算出每个expert emb到每个user emb的权重
        experts_out = torch.stack(expert_emb_list, dim=-1)
        mean_expert_emb = torch.mean(experts_out, dim=-1)
        gate_out_list = []
        for gate_linear in self.gate_linear_list:
            gate_out = gate_linear(mean_expert_emb)  # [batch,num_expert]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gate_out_list.append(gate_out)

        # 加权求和
        outs = []
        for gate_output in gate_out_list:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # [batch * 1 * num_expert]
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * emb * num_expert
            outs.append(torch.sum(weighted_expert_output, 2))  # [batch * emb]
        user_interest = torch.stack(outs, dim=1)
        return user_interest

    def forward(self, data, is_training=False):
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)
        seq_emb = self.item_emb(item_seq)
        user_emb = self.encoder(seq_emb, item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            item_e = self.item_emb(item).squeeze(1)
            cos_res = torch.bmm(user_emb, item_e)
            k_index = torch.argmax(cos_res, dim=1)

            best_interest_emb = torch.rand(user_emb.shape[0], user_emb.shape[2]).to(self.device)
            for k in range(user_emb.shape[0]):
                best_interest_emb[k, :] = user_emb[k, k_index[k], :]
            get_loss = self.cauclate_uniformity_loss(user_emb)
            loss = self.calculate_loss(best_interest_emb, item) + self.cauclate_infonce_loss(item) + get_loss * 10
            '''
            对于多兴趣任务 user_emb 需要:[batch,num_interest,emb]
            对于序列推荐 user_emb 需要:[batch,emb]
            loss 内置了多分类loss,这部分loss不需要更改，当需要额外增加loss的时候，可以在模型内部写好，只要返回依旧是loss，则外部就不需要修改
            '''
            output_dict = {
                'loss': loss,
            }
        else:
            output_dict = {
                'user_emb': user_emb,
            }
        return output_dict

class CustomModel(SequenceBaseModel):
    '''
    自定义模型，对于单纯的序列推荐/多兴趣推荐，只需要继承SequenceBaseModel写好init和forward方法即可，其他部分均可复用
    '''

    def __init__(self, enc_dict, config):
        super(CustomModel, self).__init__(enc_dict, config)

        # self.gru = torch.nn.GRU(
        #     input_size=self.embedding_dim,
        #     hidden_size=self.embedding_dim,
        #     num_layers=self.config.get('num_layers', 2),
        #     batch_first=True,
        #     bias=False
        # )
        self.encoder = BERT4RecEncoder(self.embedding_dim, self.max_length, num_layers=2, num_heads=2)
        self.reset_parameters()

    def get_aug_emb(self, item_emb):
        random_noise = torch.rand_like(item_emb, device=item_emb.device)
        item_emb = item_emb + torch.sign(item_emb) * F.normalize(random_noise, dim=-1) * self.config.get('eps', 1e-3)
        return item_emb

    def cauclate_infonce_loss(self, item):
        '''
        自定义infonce loss
        '''
        item_emb = self.output_items()
        item_emb1 = self.get_aug_emb(item_emb)
        item_emb2 = self.get_aug_emb(item_emb)

        pos_item1 = item_emb1[item]
        pos_item2 = item_emb2[item]

        pos_item1 = F.normalize(pos_item1, dim=-1)
        pos_item2 = F.normalize(pos_item2, dim=-1)
        item_emb2 = F.normalize(item_emb2, dim=-1)

        pos_score = (pos_item2 * pos_item1).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.config.get('temperature', 0.1))
        ttl_score = torch.matmul(pos_item1, item_emb2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.config.get('temperature', 0.1)).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum() * self.config.get('alpha', 1e-7)

    def forward(self, data, is_training=False):
        item_seq = data['hist_item_list']
        seq_emb = self.item_emb(item_seq)
        mask = data['hist_mask_list']
        item_seq_length = torch.sum(mask, dim=1)

        user_emb = self.encoder(seq_emb,item_seq_length)
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item)
            '''
            对于多兴趣任务 user_emb 需要:[batch,num_interest,emb]
            对于序列推荐 user_emb 需要:[batch,emb]
            loss 内置了多分类loss,这部分loss不需要更改，当需要额外增加loss的时候，可以在模型内部写好，只要返回依旧是loss，则外部就不需要修改
            '''
            output_dict = {
                'loss': loss,
            }
        else:
            output_dict = {
                'user_emb': user_emb,
            }
        return output_dict


if __name__ == '__main__':
    # 声明数据schema
    schema = {
        'user_col': 'user_id',
        'item_col': 'item_id',
        'cate_cols': ['genre'],
        'max_length': 20,
        'time_col': 'timestamp',
        'task_type': 'sequence'
    }
    # 模型配置
    config = {
        'embedding_dim': 64,
        'lr': 0.001,
        'num_layers': 2,
        'device': -1,
    }
    config['device'] = set_device(config['device'])
    config.update(schema)

    # wandb配置
    wandb_config = {
        'key': 'ca0a80eab60eff065b8c16ab3f41dec4783e60ae',
        'project': 'pangu_sequence_example',
        'name': 'exp_1',
        'config': config
    }

    # 样例数据
    train_df = pd.read_csv('./sample_data/sample_train.csv')
    valid_df = pd.read_csv('./sample_data/sample_valid.csv')
    test_df = pd.read_csv('./sample_data/sample_test.csv')

    # 声明使用的device
    device = torch.device('cpu')
    # 获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema,
                                                                       batch_size=4)
    # 声明模型,排序模型目前支持：xxx,xxx,xxx,xxx
    model = CustomModel(enc_dict=enc_dict, config=config)
    # 声明Trainer
    # trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt',wandb_config=wandb_config)
    trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt')
    # 训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device, log_rounds=10,
                use_earlystoping=True, max_patience=5, monitor_metric='recall@20', )
    # 保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './custom_model_ckpt')
    # 模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
