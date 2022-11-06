# -*- ecoding: utf-8 -*-
# @ModuleName: inference_example
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/9/19 2:44 PM
import sys
sys.path.append('../../')
import torch
from rec_pangu.models.ranking import WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM, DCN
from rec_pangu.trainer import RankTraniner
import pandas as pd
from loguru import logger


if __name__=='__main__':
    df = pd.read_csv('sample_data/ranking_sample_data.csv')
    #声明数据schema
    schema={
        "sparse_cols":['user_id','item_id','item_type','dayofweek','is_workday','city','county',
                      'town','village','lbs_city','lbs_district','hardware_platform','hardware_ischarging',
                      'os_type','network_type','position'],
        "dense_cols" : ['item_expo_1d','item_expo_7d','item_expo_14d','item_expo_30d','item_clk_1d',
                       'item_clk_7d','item_clk_14d','item_clk_30d','use_duration'],
        "label_col":'click',
    }

    # 模型ckpt地址
    model_dict = torch.load('./model_ckpt/model.pth')
    # 获取保存的enc_dict与model state dict
    enc_dict = model_dict['enc_dict']
    model_state_dict = model_dict['model']

    # 根据enc_dict初始化模型
    model = xDeepFM(enc_dict=enc_dict)
    # 读取模型的state_dict做到读取训练模型
    model.load_state_dict(model_state_dict)

    #模拟测试集
    test_df = df[:8]
    del test_df['click']

    # 声明Trainer
    trainer = RankTraniner(num_task=1)

    # #测试 predict_dataframe
    y_pre_dataftame = trainer.predict_dataframe(model, test_df, enc_dict, schema)

    logger.info('Model Inference: {}'.format(y_pre_dataftame))


