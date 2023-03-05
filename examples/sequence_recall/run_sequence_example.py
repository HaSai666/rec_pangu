# -*- ecoding: utf-8 -*-
# @ModuleName: run_sequence_example
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/3/5 18:00
import sys
sys.path.append('../../')
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.sequence import ComirecSA,ComirecDR,MIND,YotubeDNN
from rec_pangu.trainer import SequenceTrainer
from rec_pangu.utils import set_device
import pandas as pd

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
        'K': 1,
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
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema, batch_size=100)
    #声明模型,排序模型目前支持：xxx,xxx,xxx,xxx
    model = ComirecSA(enc_dict=enc_dict,config=config)
    #声明Trainer
    trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt',wandb_config=wandb_config)
    # trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt')
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device,log_rounds=10,
                use_earlystoping=True, max_patience=5, monitor_metric='recall@20',)
    #保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)