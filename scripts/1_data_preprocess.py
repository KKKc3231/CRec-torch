import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
for i in [current_dir,parent_dir,grandparent_dir]:
    if i not in sys.path:
        sys.path.append(i)
        
import json
from common.feature import feature_process, utils
from common.input import data_input_hive_spark
from aibrain_common.component import tools
from aibrain_common.utils.date_convert_utils import DateConvertUtils
import pyspark.sql.functions as F
from pyspark.sql import DataFrame as SparkDataFrame
import pandas as pd
from common.save.save_faeture_process import save_feature_process



date_converter = DateConvertUtils()
date_converter.set_biz_date("20250720")
date = date_converter.parse_data_date("${yyyymmdd}")
pre_one_day = date_converter.parse_data_date("${yyyymmdd - 1}")
pre_three_day = date_converter.parse_data_date("${yyyymmdd - 3}")
pre_six_day = date_converter.parse_data_date("${yyyymmdd - 7}")
pre_two_week = date_converter.parse_data_date("${yyyymmdd - 14}")
pre_one_month = date_converter.parse_data_date("${yyyymmdd - 30}")
pre_two_month = date_converter.parse_data_date("${yyyymmdd - 60}")
pre_three_month = date_converter.parse_data_date("${yyyymmdd - 90}")
add_two_day = date_converter.parse_data_date("${yyyymmdd + 2}")

# 配置文件
train_conf_path = './conf/train_conf.json'  # 训练配置
fg_conf_path = './conf/fg_conf.json'  # fg配置
with open(train_conf_path, 'r', encoding='utf-8') as file:
    train_conf = json.load(file)
with open(fg_conf_path, 'r', encoding='utf-8') as file:
    fg_conf = json.load(file)

def data_process(train_conf, fg_conf):

    # 规则、模型名、文件存储地址
    model_name = train_conf["output"]["model_name"]
    rule_name = train_conf["output"]["rule_name"]
    online_env_save_name = train_conf["output"]["online_env_save_name"]

    # 运行时间
    train_conf["source"]["data_source"]["train_start_time"] = pre_one_day
    train_conf["source"]["data_source"]["train_end_time"] = pre_one_day
    train_conf["source"]["data_source"]["eval_start_time"] = date
    train_conf["source"]["data_source"]["eval_end_time"] = date

    # 读取hive表数据
    train_df, test_df = data_input_hive_spark.get_data(train_conf)

    # 特征预处理
    train_data, test_data, feature_process_dict = feature_process.process(train_df, test_df, train_conf, fg_conf)
    
    
    return train_data, test_data, feature_process_dict

if __name__ == '__main__':

    train_data, test_data, feature_process_dict = data_process(train_conf, fg_conf)
    print(type(train_data))