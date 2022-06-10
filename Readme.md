# Rec PanGu
<p align="left">
  <img src='https://img.shields.io/badge/python-3.7+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.7+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-0.23.2+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5+-brightgreen'>

## 1.开源定位
- 使用pytorch对经典的rank/多任务模型进行实现，并且对外提供统一调用的API接口，极大的降低了使用Rank/多任务模型的时间成本
- 该项目使用了pytorch来实现我们的各种模型，以便于初学推荐系统的人可以更好的理解算法的核心思想
- 由于已经有了很多类似的优秀的开源，我们这里对那些十分通用的模块参考了已有的开源，十分感谢这些开源贡献者的贡献
## 2.Rank模型
这里目前支持以下Rank模型
```yaml
WDL
DeepFM
NFM
FiBiNet
AFM
AFN
AOANet
AutoInt
CCPM
LR
FM
xDeepFM
```
## 3.多任务模型
目前支持以下多任务模型
```yaml
AITM
ShareBottom
ESSM
MMOE
OMOE
MLMMOE
```
## 4.Demo
我们的Rank和多任务模型所对外暴露的接口十分相似，我们下面会分别给出Rank和多任务模型的demo
### 4.1 Rank Demo
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
这里的schema主要记录数据集的信息，主要包括离散特征的列表('sparse_cols'),连续特征列表('dense_cols'),标签列('label_cols')
### 4.2 多任务模型Demo
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
这里的schema主要记录数据集的信息，主要包括离散特征的列表('sparse_cols'),连续特征列表('dense_cols'),标签列('label_cols'),注意在多任务模型中，标签列的值为每一个任务的列名
