# -*- ecoding: utf-8 -*-
# @ModuleName: custom_model
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/6 13:46
import sys
sys.path.append('../../')
import torch
import torch.nn.functional as F
from loguru import logger
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.base_model import SequenceBaseModel
from rec_pangu.trainer import SequenceTrainer
from rec_pangu.utils import set_device
import pandas as pd

class CustomModel(SequenceBaseModel):
    '''
    自定义模型，对于单纯的序列推荐/多兴趣推荐，只需要继承SequenceBaseModel写好init和forward方法即可，其他部分均可复用
    '''
    def __init__(self, enc_dict, config):
        super(CustomModel, self).__init__(enc_dict, config)

        self.gru = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.config.get('num_layers',2 ),
            batch_first=True,
            bias=False
        )
        self.apply(self._init_weights)

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

        pos_score = (pos_item2*pos_item1).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.config.get('temperature', 0.1))
        ttl_score = torch.matmul(pos_item1, item_emb2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.config.get('temperature', 0.1)).sum(dim=1)
        return -torch.log(pos_score/ ttl_score).sum() * self.config.get('alpha', 1e-7)

    def forward(self, data, is_training=False):
        item_seq = data['hist_item_list']
        seq_emb = self.item_emb(item_seq)
        _, seq_emb = self.gru(seq_emb)
        user_emb = seq_emb[-1]
        if is_training:
            item = data['target_item'].squeeze()
            loss = self.calculate_loss(user_emb, item) + self.cauclate_infonce_loss(item)
            '''
            对于多兴趣任务 user_emb 需要:[batch,num_interest,emb]
            对于序列推荐 user_emb 需要:[batch,emb]
            loss 内置了多分类loss,这部分loss不需要更改，当需要额外增加loss的时候，可以在模型内部写好，只要返回依旧是loss，则外部就不需要修改
            '''
            output_dict = {
                'user_emb': user_emb,
                'loss': loss,
            }
        else:
            output_dict = {
                'user_emb': user_emb,
            }
        return output_dict

if __name__=='__main__':
    #声明数据schema
    schema = {
        'user_col': 'user_id',
        'item_col': 'item_id',
        'cate_cols': ['genre'],
        'max_length': 20,
        'time_col': 'timestamp',
        'task_type':'sequence'
    }
    # 模型配置
    config = {
        'embedding_dim': 64,
        'lr': 0.001,
        'num_layers': 2,
        'device':-1,
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

    #样例数据
    train_df = pd.read_csv('./sample_data/sample_train.csv')
    valid_df = pd.read_csv('./sample_data/sample_valid.csv')
    test_df = pd.read_csv('./sample_data/sample_test.csv')

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema, batch_size=4)
    #声明模型,排序模型目前支持：xxx,xxx,xxx,xxx
    model = CustomModel(enc_dict=enc_dict,config=config)
    #声明Trainer
    # trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt',wandb_config=wandb_config)
    trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt')
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device,log_rounds=10,
                use_earlystoping=True, max_patience=5, monitor_metric='recall@20',)
    #保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './custom_model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)