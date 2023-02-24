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
这里目前支持以下Rank模型

| 模型     | 论文                                                                                                                                                                                                                                                                                                             | 年份   | 相关资料 |
|--------|------|------|------|
| WDL    | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)     | 2016 | TBD  |
| DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247)     | 2017 | TBD  |
| NFM | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)              | 2017 | TBD  |
| FiBiNet | [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate](https://arxiv.org/pdf/1905.09433.pdf) | 2019 | TBD  |
| AFM | [Attentional Factorization Machines](https://arxiv.org/pdf/1708.04617)                  | 2017 | TBD  |
| AutoInt | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)              | 2018 | TBD  |
| CCPM | [A Convolutional Click Prediction Model](http://www.shuwu.name/sw/Liu2015CCPM.pdf)    | 2015 | TBD  |
| LR | /  | 2019 | TBD  |
| FM | /  | 2019 | TBD  |
| xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)     | 2018 | TBD  |
| DCN | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf) | 2019 | TBD  |

## 4.多任务模型
目前支持以下多任务模型

| 模型          | 论文                                                                                                                                          | 年份   | 相关资料 |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| MMOE        | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 | TBD  |
| ShareBottom | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 | TBD  |
| ESSM        | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)      | 2018 | TBD  |
| OMOE        | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | 2018 | TBD  |
| MLMMOE      | /                                                                                                                                           | /    | TBD  |
| AITM        | [Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising](https://arxiv.org/pdf/2105.08489.pdf)| 2019 | TBD  |

## 5.Demo
我们的Rank和多任务模型所对外暴露的接口十分相似,同时我们这里也支持使用wandb来实时监测模型训练指标,我们下面会分别给出Rank,多任务模型,wandb的demo
### 5.1 Rank Demo
```python
#声明数据schema
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.ranking import WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM
from rec_pangu.trainer import RankTraniner
import pandas as pd

if __name__=='__main__':
    df = pd.read_csv('sample_data/ranking_sample_data.csv')
    print(df.head())
    #声明数据schema
    schema={
        "sparse_cols":['user_id','item_id','item_type','dayofweek','is_workday','city','county',
                      'town','village','lbs_city','lbs_district','hardware_platform','hardware_ischarging',
                      'os_type','network_type','position'],
        "dense_cols" : ['item_expo_1d','item_expo_7d','item_expo_14d','item_expo_30d','item_clk_1d',
                       'item_clk_7d','item_clk_14d','item_clk_30d','use_duration'],
        "label_col":'click',
    }
    #准备数据,这里只选择了100条数据,所以没有切分数据集
    train_df = df
    valid_df = df
    test_df = df

    #声明使用的device
    device = torch.device('cpu')
    #获取dataloader
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema)
    #声明模型,排序模型目前支持：WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM
    model = xDeepFM(enc_dict=enc_dict)
    #声明Trainer
    trainer = RankTraniner(num_task=1)
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=5, lr=1e-3, device=device)
    #保存模型权重
    trainer.save_model(model, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
    print('Test metric:{}'.format(test_metric))

```
### 5.2 多任务模型Demo
```python
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.multi_task import AITM,ShareBottom,ESSM,MMOE,OMOE,MLMMOE
from rec_pangu.trainer import RankTraniner
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
    train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema)
    #声明模型,多任务模型目前支持：AITM,ShareBottom,ESSM,MMOE,OMOE,MLMMOE
    model = AITM(enc_dict=enc_dict)
    #声明Trainer
    trainer = RankTraniner(num_task=2)
    #训练模型
    trainer.fit(model, train_loader, valid_loader, epoch=5, lr=1e-3, device=device)
    #保存模型权重
    trainer.save_model(model, './model_ckpt')
    #模型验证
    test_metric = trainer.evaluate_model(model, test_loader, device=device)
    print('Test metric:{}'.format(test_metric))
```
### 5.3 Wandb Demo
```python
import torch
from rec_pangu.dataset import get_dataloader
from rec_pangu.models.ranking import WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM, DCN
from rec_pangu.trainer import RankTraniner
import pandas as pd
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
# 只需要额外增加wandb_config即可
wandb_config = {
    'key':'ca0a80eab60eff065b8c16ab3f41dec4783e60ae',
    'project':'pangu_ranking_example',
    'name':'exp_2',
    'config':{
        'embedding_dim':16,
        'hidden_units':[64,32,16]
    }
}
#准备数据,这里只选择了100条数据,所以没有切分数据集
train_df = df[:80]
valid_df = df[:90]
test_df = df[:95]

#声明使用的device
device = torch.device('cpu')
#获取dataloader
train_loader, valid_loader, test_loader, enc_dict = get_dataloader(train_df, valid_df, test_df, schema, batch_size=512)
#声明模型,排序模型目前支持：WDL, DeepFM, NFM, FiBiNet, AFM, AFN, AOANet, AutoInt, CCPM, LR, FM, xDeepFM
model = DeepFM(**wandb_config['config'],enc_dict=enc_dict)
#声明Trainer
trainer = RankTraniner(num_task=1,wandb_config=wandb_config)
#训练模型
trainer.fit(model, train_loader, valid_loader, epoch=500, lr=1e-3, device=device,
            use_earlystoping=True, max_patience=5, monitor_metric='valid_roc_auc_score')
#保存模型权重
# trainer.save_model(model, './model_ckpt')
#保存模型权重和enc_dict
trainer.save_all(model, enc_dict, './model_ckpt')
#模型验证
test_metric = trainer.evaluate_model(model, test_loader, device=device)
```