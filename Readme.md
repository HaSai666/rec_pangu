# Rec PanGu
[![stars](https://img.shields.io/github/stars/HaSai666/rec_pangu?color=097abb)](https://github.com/HaSai666/rec_pangu/stargazers)
[![issues](https://img.shields.io/github/issues/HaSai666/rec_pangu?color=097abb)](https://github.com/HaSai666/rec_pangu/issues)
[![license](https://img.shields.io/github/license/HaSai666/rec_pangu?color=097abb)](https://github.com/HaSai666/rec_pangu/blob/main/LICENSE)
<img src='https://img.shields.io/badge/python-3.7+-brightgreen'>
<img src='https://img.shields.io/badge/torch-1.7+-brightgreen'>
<img src='https://img.shields.io/badge/scikit_learn-0.23.2+-brightgreen'>
<img src='https://img.shields.io/badge/pandas-1.0.5+-brightgreen'>
<img src='https://img.shields.io/badge/pypi-0.2.4-brightgreen'>
<a href="https://wakatime.com/badge/user/4f5f529d-94ee-4a12-94de-38e886b0219b/project/5b1e1c2d-5596-4335-937e-f2b5515a7fab"><img src="https://wakatime.com/badge/user/4f5f529d-94ee-4a12-94de-38e886b0219b/project/5b1e1c2d-5596-4335-937e-f2b5515a7fab.svg" alt="wakatime"></a>
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/27fce22d928f412bb7cb3dab813ab481)](https://app.codacy.com/gh/HaSai666/rec_pangu/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Downloads](https://pepy.tech/badge/rec-pangu)](https://pepy.tech/project/rec-pangu)
[![Downloads](https://pepy.tech/badge/rec-pangu/month)](https://pepy.tech/project/rec-pangu)
[![Downloads](https://pepy.tech/badge/rec-pangu/week)](https://pepy.tech/project/rec-pangu)
## 1.开源定位 
- 使用pytorch对经典的rank/多任务模型进行实现，并且对外提供统一调用的API接口，极大的降低了使用Rank/多任务模型的时间成本
- 该项目使用了pytorch来实现我们的各种模型，以便于初学推荐系统的人可以更好的理解算法的核心思想
- 由于已经有了很多类似的优秀的开源，我们这里对那些十分通用的模块参考了已有的开源，十分感谢这些开源贡献者的贡献 
## 2.安装  
```bash
#最新版
git clone https://github.com/HaSai666/rec_pangu.git
cd rec_pangu
pip install -e . --verbose

#稳定版 
pip install rec_pangu --upgrade
```
## 3.Rank模型

| 模型      | 论文     | 年份   | 
|---------|------|------|
| WDL     | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)     | 2016 | 
| DeepFM  | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247)     | 2017 | 
| NFM     | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)              | 2017 | 
| FiBiNet | [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate](https://arxiv.org/pdf/1905.09433.pdf) | 2019 | 
| AFM     | [Attentional Factorization Machines](https://arxiv.org/pdf/1708.04617)                  | 2017 | 
| AutoInt | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)              | 2018 | 
| CCPM    | [A Convolutional Click Prediction Model](http://www.shuwu.name/sw/Liu2015CCPM.pdf)    | 2015 |
| LR      | /  | 2019 | 
| FM      | /  | 2019 | 
| xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)     | 2018 |
| DCN     | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf) | 2019 | 
| MaskNet | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/pdf/2102.07619.pdf) | 2021 | 
## 4.多任务模型

| 模型          | 论文                                                                                                                                          | 年份   | 
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------|------|
| MMOE        | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 |
| ShareBottom | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 |
| ESSM        | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)      | 2018 |
| OMOE        | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 |
| MLMMOE      | /                                                                                                                                           | /    |
| AITM        | [Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising](https://arxiv.org/pdf/2105.08489.pdf)| 2019 |

## 5.序列召回模型

目前支持如下类型的序列召回模型:

- 经典序列召回模型
- 基于图的序列召回模型
- 多兴趣序列召回模型


| 模型        | 类型     | 论文                                                                                                                                     | 年份   | 
|-----------|--------|-------------------------------------------------------------------------------------------------------------------------------------------|------|
| YotubeDNN | 经典序列召回 |[Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190?utm_campaign=Weekly%20dose%20of%20Machine%20Learning&utm_medium=email&utm_source=Revue%20newsletter) | 2016 |
| Gru4Rec   | 经典序列召回 |[Session-based recommendations with recurrent neural networks](https://arxiv.org/pdf/1511.06939) | 2015 |
| Narm      | 经典序列召回 |[Neural Attentive Session-based Recommendation](https://arxiv.org/pdf/1711.04725) | 2017 |
| NextItNet | 经典序列召回 |[A Simple Convolutional Generative Network for Next Item](https://arxiv.org/pdf/1808.05163) | 2019 |
| ComirecSA | 多兴趣召回  |[Controllable multi-interest framework for recommendation](https://arxiv.org/pdf/2005.09347) | 2020 |
| ComirecDR | 多兴趣召回  |[Controllable multi-interest framework for recommendation](https://arxiv.org/pdf/2005.09347) | 2020 |
| Mind      | 多兴趣召回  |[Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030) | 2019 |
| Re4       | 多兴趣召回  |[Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation](https://dl.acm.org/doi/10.1145/3485447.3512094) | 2022 |
| CMI       | 多兴趣召回  |[mproving Micro-video Recommendation via Contrastive Multiple Interests](https://arxiv.org/pdf/2205.09593) | 2022 |
| SRGNN     | 图序列召回  |[Session-based Recommendation with Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/3804/3682) | 2019 |
| GC-SAN    | 图序列召回  |[SGraph Contextualized Self-Attention Network for Session-based Recommendation](https://www.ijcai.org/proceedings/2019/0547.pdf) | 2019 |
| NISER     | 图序列召回  |[NISER: Normalized Item and Session Representations to Handle Popularity Bias](https://arxiv.org/pdf/1909.04276) | 2019 |



## 6.图协同过滤模型

TODO

## 7.Demo
我们的Rank和多任务模型所对外暴露的接口十分相似,同时我们这里也支持使用wandb来实时监测模型训练指标,我们下面会分别给出Rank,多任务模型,wandb的demo
### 7.1 排序任务Demo

```python
# 声明数据schema
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.ranking import WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM
from rec_pangu.trainer import RankTrainer
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('sample_data/ranking_sample_data.csv')
    print(df.head())
    # 声明数据schema
    schema = {
        "sparse_cols": ['user_id', 'item_id', 'item_type', 'dayofweek', 'is_workday', 'city', 'county',
                        'town', 'village', 'lbs_city', 'lbs_district', 'hardware_platform', 'hardware_ischarging',
                        'os_type', 'network_type', 'position'],
        "dense_cols": ['item_expo_1d', 'item_expo_7d', 'item_expo_14d', 'item_expo_30d', 'item_clk_1d',
                       'item_clk_7d', 'item_clk_14d', 'item_clk_30d', 'use_duration'],
        "label_col": 'click',
    }
    # 准备数据,这里只选择了100条数据,所以没有切分数据集
    train_df = df
    valid_df = df
    test_df = df

    # 声明使用的device
    device = torch.device('cpu')
    # 获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema)
    # 声明模型,排序模型目前支持：WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM
    model = xDeepFM(enc_dict=enc_dict)
    # 声明Trainer
    trainer = RankTrainer(num_task=1)
    # 训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=5, lr=1e-3, device=device)
    # 保存模型权重
    trainer.save_model(model, './model_ckpt')
    # 模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
    print('Test metric:{}'.format(test_metric))

```
### 7.2 多任务模型Demo

```python
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.multi_task import AITM, ShareBottom, ESSM, MMOE, OMOE, MLMMOE
from rec_pangu.trainer import RankTrainer
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('sample_data/multi_task_sample_data.csv')
    print(df.head())
    #声明数据schema
    schema = {
        "sparse_cols": ['user_id', 'item_id', 'item_type', 'dayofweek', 'is_workday', 'city', 'county',
                        'town', 'village', 'lbs_city', 'lbs_district', 'hardware_platform', 'hardware_ischarging',
                        'os_type', 'network_type', 'position'],
        "dense_cols": ['item_expo_1d', 'item_expo_7d', 'item_expo_14d', 'item_expo_30d', 'item_clk_1d',
                       'item_clk_7d', 'item_clk_14d', 'item_clk_30d', 'use_duration'],
        "label_col": ['click', 'scroll'],
    }
    #准备数据,这里只选择了100条数据,所以没有切分数据集
    train_df = df
    valid_df = df
    test_df = df

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema)
    #声明模型,多任务模型目前支持：AITM,ShareBottom,ESSM,MMOE,OMOE,MLMMOE
    model = AITM(enc_dict=enc_dict)
    #声明Trainer
    trainer = RankTrainer(num_task=2)
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=5, lr=1e-3, device=device)
    #保存模型权重
    trainer.save_model(model, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
    print('Test metric:{}'.format(test_metric))
```
### 7.3 序列召回Demo
```python
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.sequence import ComirecSA,ComirecDR,MIND,CMI,Re4,NARM,YotubeDNN,SRGNN
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

    #样例数据
    train_df = pd.read_csv('./sample_data/sample_train.csv')
    valid_df = pd.read_csv('./sample_data/sample_valid.csv')
    test_df = pd.read_csv('./sample_data/sample_test.csv')

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema, batch_size=50)
    #声明模型,序列召回模型模型目前支持： ComirecSA,ComirecDR,MIND,CMI,Re4,NARM,YotubeDNN,SRGNN
    model = ComirecSA(enc_dict=enc_dict,config=config)
    #声明Trainer
    trainer = SequenceTrainer(model_ckpt_dir='./model_ckpt')
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device,log_rounds=10,
                use_earlystoping=True, max_patience=5, monitor_metric='recall@20',)
    #保存模型权重和enc_dict
    trainer.save_all(model, enc_dict, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)

```
### 7.4 图协调过滤Demo
TODO