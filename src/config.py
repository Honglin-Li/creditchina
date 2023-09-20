# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

# configs  
s_year = 2014
e_year = 2022
e_date = "2022-12-31"

usci = "统一社会信用代码 Unified Social Credit Identifier, USCI"
company_name = "机构名称 Institution name"

#data_path = "X:\\ProjectData\\SCS\\creditchina\\data\\" # switch to this one if you need to use jupyter
data_path = os.path.join(os.getcwd(), 'data')

# path variables
sub_event_path = os.path.join(data_path, 'clean_sub_events')
original_path = os.path.join(data_path, 'original')
attri_path = os.path.join(data_path, 'company_attributes')

stat_path = os.path.join(data_path, 'stat')

processed_data_path = os.path.join(data_path, 'processed_data')
processed_event_path = os.path.join(processed_data_path, 'events')
processed_meta_event_path = os.path.join(processed_data_path, 'meta_events')
performance_path = os.path.join(processed_data_path, 'performance_panel')

# check path exists
path_l = [data_path,
          sub_event_path, 
          original_path,
          processed_data_path, 
          processed_event_path, 
          processed_meta_event_path,
          performance_path,
          stat_path]

for p in path_l:
    if not os.path.exists(p):
        os.makedirs(p)

