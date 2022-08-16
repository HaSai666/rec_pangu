# -*- ecoding: utf-8 -*-
# @ModuleName: run_multi_task_benchmark_example
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/13 5:02 PM
import sys
sys.path.append('../../')
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.benchmark_trainer import BenchmarkTrainer
import pandas as pd


if __name__=='__main__':
    df = pd.read_csv('sample_data/multi_task_sample_data.csv')
    print(df.head())
    #声明数据schema
    schema={
        "sparse_cols":['user_id','item_id','item_type','dayofweek','is_workday','city','county',
                      'town','village','lbs_city','lbs_district','hardware_platform','hardware_ischarging',
                      'os_type','network_type','position'],
        "dense_cols" : ['item_expo_1d','item_expo_7d','item_expo_14d','item_expo_30d','item_clk_1d',
                       'item_clk_7d','item_clk_14d','item_clk_30d','use_duration'],
        "label_col":['click','scroll'],
    }
    #准备数据,这里只选择了100条数据,所以没有切分数据集
    train_df = df
    valid_df = df
    test_df = df

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema, batch_size=512)
    #声明需要跑测的模型
    model_list = ['MMOE','AITM','ShareBottom','ESSM','OMOE','MLMMOE']
    # 声明Benchmark Trainer
    benchmark_trainer = BenchmarkTrainer(num_task=2,
                                         model_list=model_list,
                                         benhcmark_res_path='./multi_task_benchmark_res.csv',
                                         ckpt_root='./multi_task_benchmark_ckpt/')
    #开始benchmark跑测，模型权重保存在{ckpt_root}/{model_name}下面，benchmark的输出结果保存在benchmark_res_path里面
    benchmark_trainer.run(enc_dict=enc_dict,
                          train_loader=train_loader,
                          valid_loader=valid_loader,
                          test_loader=test_loader,
                          epoch=3,
                          lr=1e-3,
                          device=device
                          )