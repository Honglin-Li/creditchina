#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
event5_processing

Processing event5 commitment.
"""

# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# df_event = pd.read_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'))

# # manual check those rows without region info
# auth_col = '承诺受理单位 Commitment processing unit'
# 
# t = extract_authorites(df_event[auth_col]).join(df_event[auth_col])
# 
# # show the rows without region, to identify posibble region dict
# s_auth = t.loc[t.level=='uncertain', auth_col].value_counts()
# 
# print(f'unknown records: {s_auth.sum()}')
# print(f'unique unknown records: {s_auth.shape}')
# 
# # check 50 one time
# s_auth[:50]

# In[2]:


def add_regions(df_event):
    """
    Add region related columns to df_event.
    
    This function is implemented by calling :func:`utils.extract_authorites`.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        df_event with region columns.
    """
    # some authorities have no 省 市 区ending, finding the dict by manual checking the rows where level='uncertain'
    replace_dict = {
        '榆阳': '陕西省榆林市榆阳区',
        '许继': '河南省许昌市许继',
        '仙人岛': '辽宁省营口市盖州市仙人岛',
        '平煤隆基新能源科技有限公司': '河南省许昌市襄城县平煤隆基新能源科技有限公司',
        '水源镇': '甘肃省金昌市永昌县水源镇',
        '上清寺': '重庆市渝中区上清寺',
        '甘孜州': '四川省甘孜藏族自治州',
        '中纺院绿色纤维股份公司': '河南省新乡市中纺院绿色纤维股份公司',
        '柿铺街道': '湖北省襄阳市樊城区柿铺街道',
        '王寨街道': '湖北省襄阳市樊城区王寨街道',
        '七星岗街道': '重庆市渝中区七星岗街道',
        '鼓浪屿': '福建省厦门市思明区鼓浪屿',
        '清河口街道': '湖北省襄阳市樊城区清河口街道',
        '化龙桥街道': '重庆市渝中区化龙桥街道',
        '上清寺街道': '重庆市渝中区上清寺街道',
        '中国有色金属工业第六冶金建设有限公司': '河南省郑州市中原区中国有色金属工业第六冶金建设有限公司',
        '淄川': '山东省淄博市淄川区',
        '菜园坝街道': '重庆市渝中区菜园坝街道',
        '解放碑街道': '重庆市渝中区解放碑街道',
        '汉江街道': '湖北省襄阳市樊城区汉江街道',
        '上蔡牧原农牧有限公司': '河南省南阳市上蔡牧原农牧有限公司',
        '太平店镇': '湖北省襄阳市樊城区太平店镇',
        '襄城': '河南省许昌市襄城县',
        '古港镇': '湖南省浏阳市古港镇',
        '延津': '河南省新乡市延津县',
        '汝南': '河南省驻马店市汝南县',
        '长垣': '河南省长垣市',
        '辛店': '山东省淄博市临淄区辛店',
        '大溪沟街道': '重庆市渝中区大溪沟街道'
    }
    
    s_auth = \
    df_event['承诺受理单位 Commitment processing unit'].replace(r'[a-zA-Z\d]+', '', regex=True).replace(replace_dict, regex=True)
    
    # get region info
    df_region = extract_authorites(s_auth)

    # append to original df: province, city, area, level
    return df_event.join(df_region) 


# In[3]:


def process_event():
    """
    Add region columns to df_event, then get cross tables.
    
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
    # load event
    print('load data...')
    df_event = pd.read_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'))
    
    # rename
    df_event.rename(columns={
        '承诺类型 Commitment type': 'commit_type',
        '承诺事由 Commitment reason': 'commit_reason'
    }, inplace=True)
    
    # translate commit type
    commit_type_map = {
            '主动型': 'Proactive',
            '审批替代型': 'Approval substitution',
            '证明事项型': 'Certification matters',
            '行业自律型': 'Industry self-regulation',
            '信用修复型': 'Credit repair'
        }

    df_event.commit_type = df_event.commit_type.replace(commit_type_map)
    
    print('add columns: region, law, fine amount, type...')
    
    df_event = df_event.pipe(add_regions)
    
    # handle unknown provinces
    df_event = standard_province_names(df_event)
    
    draw_distribution(
        [df_event.start_year, df_event.province, df_event.level, df_event.commit_type],
        ['bar', 'pie', 'pie', 'pie']
    )

    print('save processed df_event...')
    
    df_event[['company_name', 
               'theme',
               'province',
               'level',
               'authority',
               'commit_type',
               'commit_reason',
               'start_year',
               'start_date'
              ]].to_excel(os.path.join(processed_event_path, 'event5.xlsx'), 
                          index=False, 
                          freeze_panes=(1, 1))

    print('generate cross tables...')
    
    # set parameters
    cat_column_names = ['level', 'theme', 'commit_type']
    cat_prefix = ['E5_Auth_level:', 'E5_Theme:', 'E5_Commit_Type:']
    number_column_name = 'commit'
    multi_flags = [False, False, False]
    save_prefix = 'event5'
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
    


# process_event()

# In[ ]:




