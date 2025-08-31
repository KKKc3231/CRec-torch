import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
for i in [current_dir,parent_dir,grandparent_dir]:
    if i not in sys.path:
        sys.path.append(i)

import json
from aibrain_common.utils.date_convert_utils import DateConvertUtils
from aibrain_common.data.dataset_builder import (ColumnSpec, DatasetBuilder)
from aibrain_job.utils import param_utils
from common import utils
from common.save.save_faeture_process import save_feature_process
import pandas as pd
import numpy as np
from aibrain_common.component import tools
from common.feature.fg_parser import FgParser
from TorchDatasetBuilder import TorchDatasetAdapter

date_converter = DateConvertUtils()
date_converter.set_biz_date("20250712")
date = date_converter.parse_data_date("${yyyymmdd}")

# 配置文件

train_conf_path = os.path.join(current_dir, "conf", "train_conf.json")  # 训练配置
fg_conf_path = os.path.join(current_dir, "conf", "fg_conf.json")  # fg配置

with open(train_conf_path, 'r', encoding='utf-8') as file:
    train_conf = json.load(file)
with open(fg_conf_path, 'r', encoding='utf-8') as file:
    fg_conf = json.load(file)


# fg_conf处理
fg_conf_parser = FgParser(fg_conf)

# 构建input_fn
local_file_path = os.path.join(current_dir, "upload_file", "RcarCarTypeModelRecomFeatureProcess_seq.json")

with open( local_file_path, 'r', encoding='utf8') as fp:
    feature_process_dict = json.load(fp)

spec = utils.featuers_spec_from_fg(fg_conf_parser)

test_dataset_builder = DatasetBuilder(
        input_table=train_conf["source"]["data_source"]["table_name"] + "_seq_train_cs_708_711",
        partitions=f'pt={date}', column_spec=spec
)

df_train = test_dataset_builder.to_pandas()
df_train.to_csv("708_711_train.csv")