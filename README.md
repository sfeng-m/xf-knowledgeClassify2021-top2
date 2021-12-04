# xf-knowledgeClassify2021-top2
2021科大讯飞试题标签预测挑战赛亚军方案



环境：python3.6 + pytorch1.7
基于pytorch1.7环境，载入相关的安装包: "pip install -r requirements.txt"

模型加载：
将百度网盘中的“save_model”文件夹存放于主目录下；（链接:https://pan.baidu.com/s/1nr6nsB5Qsm32MrMbaQUdww  密码:objk）

执行方法：

1. 模型训练：sh train.sh
2. 模型预测：sh test.sh

两点说明：

1. 本方案只使用官方提供的数据集，未使用任何外部数据；
2. 本方案的预训练模型为roberta_chinese_wwm_ext, 可在huggingface中进行下载。
