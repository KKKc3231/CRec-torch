# CRec-torch🚗
pytorch从0-1实现推荐算法，ctr / cvr &amp;&amp; seqrec &amp;&amp; generative rec

在实习期间，重构的一套pytorch的推荐模型训练流程，包括了常见的ctr / cvr 多任务点击转化预测任务，序列推荐任务（基于用户交互item_id的先后顺序），生成式训练范式实现序列推荐

- 规划实现中～
- 1、ctr / cvr 全流程已实现，多任务架构，特征处理 / 数据加载 / 模型训练 / 推理
- 2、seqrec：sasrec / hstu decoder-only 已复现
- 3、generative rec：生成式推荐（偏recall），基本复现，规划中～
