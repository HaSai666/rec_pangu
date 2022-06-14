# [Rec PanGu](https://github.com/HaSai666/rec_pangu)

  <img src='https://img.shields.io/badge/python-3.7+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.7+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-0.23.2+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5+-brightgreen'>
  <img src='https://img.shields.io/badge/pypi-0.0.3-brightgreen'>


## 开源定位
- 使用pytorch对经典的rank/多任务模型进行实现，并且对外提供统一调用的API接口，极大的降低了使用Rank/多任务模型的时间成本
- 该项目使用了pytorch来实现我们的各种模型，以便于初学推荐系统的人可以更好的理解算法的核心思想
- 由于已经有了很多类似的优秀的开源，我们这里对那些十分通用的模块参考了已有的开源，十分感谢这些开源贡献者的贡献

## 核心特点
- 我们提供了十分易用的API来完成Rank/多任务模型的使用
- 我们提供了一种简易的特征使用的方案，仅输入schema即可完成数据层面的建模
- 新增了Benchmark功能，支持在特征数据集上进行Benchmark实验

## TODO
- 增加每个模型的超参数优化功能
- 新增更多的模型