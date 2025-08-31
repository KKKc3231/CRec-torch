# 代码结构

- src这个目录是ctr cvr预测任务的一些处理的逻辑
- spark2pd.py: 目前需要将spark先转换为pandas，然后再处理数据
- dataprocess.py处理数据，归一化以及类别的encoder映射
- dataset.py：对处理后的数据，通过dataset来得到torch的dataset，从而送入到dataloader
- TorchDatasetBuilder.py: 本来是直接在aibrain的源码重构，这样效率高一些，有点问题目前