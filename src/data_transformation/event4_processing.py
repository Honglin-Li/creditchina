#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
event4_processing

Processing event4 watchlist.

Contents:
    INPUT:
    ---
    - clean_sub_events/4_event_abnormal_operations



    OUTPUT:
    ---
    - processed_data/events/4_event_abnormal_operations
        - ID
        - company_name
        - start_year
        - end_year
        - inclusion_reason
            - A. failure to publicize the annual report (in accordance with the "Interim Regulations on Enterprise Information Disclosure", Article VIII.)
            - B. Failure to publicize enterprise information within the ordered period: failure to disclose information about the enterprise within the period ordered by the administrative department for industry and commerce in accordance with the "Interim Regulations on the Public Disclosure of Enterprise Information" Article X.
            - C. Falsehoods in corporate public information
            - D. Can not be contacted through the registered residence or place of business
    - processed_data/meta_events/crosstab_event4
        - cross table for company and categoreis
    - processed_data/crosstab_year_event4
        - cross table for company, year and categoreis
    - processed_data/crosstab_day_event4


    PROCESS:
    ---
    - custom functions to extract region, level, province from authorities and regions.xlsx `extract_region_from_authority(region_s)`
    - extract inclusion reason(by the official document)
    - crosstab on the table to get stats
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# # add region columns 
# 
# - pure_region
# - level
# - province
# - city
# - area

# In[2]:


def add_regions(df_event):
    """
    Add region related columns to df_event.
    
    This function is implemented by calling :func:`utils.extract_regions`.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        df_event with region columns.
    """
    # some authorities have no 省 市 区ending
    replace_dict = {
        '锦江': '锦江区',
        '大柴旦': '大柴旦行政区',
        '成华': '成华区',
        '甘孜州': '甘孜藏族自治州'
    }
    
    df_event['列入决定机关名称 Listing decision authority name'] = \
    df_event['列入决定机关名称 Listing decision authority name'].replace(replace_dict, regex=True)
    
    # remove authorities
    auth_pattern = r'市场.+|工商.+|管委会.+|管理委员.+|行政审批.+|食品药品.+|分局.*|行委.*|审批.*|能源化工.*|生产建设.*'
    df_event['pure_region'] = df_event['列入决定机关名称 Listing decision authority name'].str.replace(
        auth_pattern, '', regex=True)
    
    # get region info
    df_region = extract_regions(df_event.pure_region)[['province', 'city', 'area', 'level']]

    # append to original df: province, city, area, level
    return df_event.join(df_region) 


# # add inclusion reason column

# In[3]:


def add_inclusion_reasons(df_event):
    """
    Add category column: inclusion_reason.
    
    The 6 reasons are from official documents.
    This func is implemented by calling :func:`utils.add_single_choice_category_from_keywords`.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        df_event with inclusion_reason column.
    """
    inclusion_reason_dict = {
        '年度报告':'A. Failure to publicize the annual report',
        '公示年报':'A. Failure to publicize the annual report',
        '有关企业信息':'B. Failure to publicize enterprise information within the ordered period',
        '弄虚作假':'C. Falsehoods in corporate public information',
        '住所':'D. Can not be contacted through the registered residence or place of business',
        '— —': 'NA'
    }
    
    return add_single_choice_category_from_keywords(
        df_event, 
        'inclusion_reason',
        '列入经营异常名录原因类型名称 Reason type for listing in Operational abnormality',
        inclusion_reason_dict)


# # call functions in order

# In[4]:


def process_event():
    """
    Add inclusion_reason category column & region columns to df_event, then get cross tables.
    
    This func add category columns to df_event, then save processed df_event to local. 
    Then generate, save, and return cross tables of company and categories and cross tables of [company, year] and categoreis.
    
    Returns
    -------
    df_crosstab_event : DataFrame
        Cross tables of company and categories.
    df_crosstab_event_year : DataFrame
        Cross tables of [company, year] and categoreis.
    df_period : DataFrame
        Data from each time point.
    """
    print('load data...')
    df_event = pd.read_excel(os.path.join(sub_event_path, '4_event_abnormal_operations.xlsx'))
    
    print('add 1. region & 2. inclusion_reason...')
    
    df_event = df_event.pipe(add_inclusion_reasons).pipe(add_regions)
    
    # handle unknown provinces
    df_event = standard_province_names(df_event)

    draw_distribution(
        [df_event.start_year, df_event.province, df_event.level],
        ['bar', 'pie', 'pie']
    )

    print('save processed df_event...')
    
    df_event[['company_name', 
               'inclusion_reason',
               'pure_region',
               'province',
               'level',
               'start_year',
               'start_date', 
               '列入决定机关名称 Listing decision authority name', 
              ]].rename(
                    columns={'列入决定机关名称 Listing decision authority name': 'authority'}
                ).to_excel(os.path.join(processed_event_path, 'event4.xlsx'), 
                          index=False, 
                          freeze_panes=(1, 1))

    
    print('generate cross tables of campany and categories...')

    cat_column_names = ['level', 'inclusion_reason']
    cat_prefix = ['E4_Auth_level:', 'E4_Incl_reason:']
    number_column_name = 'watchlist'
    multi_flags = [False, False]
    save_prefix = 'event4'
    start_year_col = 'start_year'
    start_date_col = 'start_date'

    return create_all_crosstabs(df_event,
                                cat_prefix, 
                                cat_column_names, 
                                number_column_name, 
                                multi_flags,
                                save_prefix,
                                start_year_col,
                                start_date_col
                                )
    


# df_crosstab_event, df_crosstab_event_year, df_period = process_event()
# 

# In[ ]:




