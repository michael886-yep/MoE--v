singel expert文件夹是使用的四个单专家模型，包括LSTM，LSTM-Sequence2Sequence，Transformer和LSTM-Transformer

top-1 moe文件夹是常规的top-1模型，里面尝试了使用三种模型作为路由分类器，分别是RF，XGboost和LightGBM

two-layer moe文件夹内部包含基于流量分类的双层MoE的处理过程，第一层分类器仅尝试了XGboost，第二层分类器则同样尝试了Top-1 MoE中
的三种路由器类型。文件夹中的代码表示了研究中的处理过程。先通过流量XGboost.py进行第一层的流量分类，划分为高中低流量；再经过高中低流量
数据集重构.py，再经过导入类别.py，然后再经过第二层低/中/高MoE.py进行分类。根据id选出t.py、通过数字选模型.py，以及高中低指标计算.py及最终合并.py这几个
文件均用于处理结果并计算相应指标

dynamic moe文件夹内部包含动态MoE的处理过程。先经过排列分类确定类别，再分别通过RF高中低.py文件和模型RF概率.py文件输出残差类别和
各个专家模型被选择的概率，替代人工.py和前n个模型.py则用于整合结果。
