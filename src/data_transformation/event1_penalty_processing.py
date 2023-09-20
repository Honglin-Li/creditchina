#!/usr/bin/env python
# coding: utf-8

# In[34]:


"""
event1_penalty_processing

Processing event penalty.
"""

# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# df_event = pd.read_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'), dtype={
#         'start_year': int
#     })

# In[40]:


def add_penalty_law(df_event):
    """
    Add pure law column to event table.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        Contains column: penalty_law
    """
    # extract law
    df_event['penalty_law'] = (df_event['处罚依据 Penalty Basis'] + 
                                       df_event['违法行为类型 Type of Illegal Behavior']).map(extract_law)

    # clean
    df_event['penalty_law'] = df_event['penalty_law'].str.replace(r'[《》、;&a-zA-Z\d]+', '', regex=True)
    
    # remove region info
    df_event['penalty_law'] = df_event['penalty_law'].str.replace(r'\A(中华人民共和国)?(.*省)?(.*市)??', '', regex=True)
    
    # stat
    df_event['penalty_law'].value_counts().to_excel(os.path.join(data_path, 'penalty_laws.xlsx'))
    # original 7825 rows-> roughly 773 different laws. TODO category into diff situations
    
    return df_event


# In[41]:


def add_fine(df_event):
    """
    Add fine amount to event table.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        Contains column: fine
    """
    df_event['penalty_10K'] = df_event[
            '罚款金额(万元) Fine Amount, 10k Yuan'
        ] + df_event['没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan']
    
    return df_event


# In[54]:


def get_penalty_type(row):
    """
    Extract penalty type.
    5 types can be extracted: '其他','吊销暂扣','没收','警告','罚款'. One record can have multiple types.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        Contains column: fine
    """
    # multiple choice
    types = []
    
    p_type = str(row['处罚类别 Penalty Type'])
    fine = row['罚款金额(万元) Fine Amount, 10k Yuan']
    il_gain = row['没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan']
    cert = str(row['暂扣或吊销证照名称及编号 Suspension or Revocation of License Name and Number'])
    
    # check penalty type text
    if '罚款' in p_type:
        types.append('Fine')
    if '警告' in p_type:
        types.append('Warning')
    if '没收' in p_type:
        types.append('Confiscation illegal gains')
    if ('吊销' in p_type) or ('暂扣' in p_type):
        types.append('Suspension revocation permission')
    
    # check fine column
    if (fine > 0) & ('Fine' not in types):
        types.append('Fine')
    
    # check confiscation column
    if (il_gain > 0) & ('Confiscation illegal gains' not in types):
        types.append('Confiscation illegal gains')
    
    # check supension column
    if (len(cert) > 3) & ('Suspension revocation permission' not in types):
        types.append('Suspension revocation permission')
    
    # others
    if len(types) == 0:
        types.append('Others')
        
    return types

def add_penalty_type(df_event):
    """
    Add penalty type to event table.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        Contains column: fine
    """
    df_event['penalty_type'] = df_event.apply(get_penalty_type, axis=1)
    
    return df_event
    


# # manual check those rows without region info
# auth_col = '处罚机关 Penalty Enforcement Authority'
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

# In[43]:


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
        '黔南州': '贵州省黔南布依族苗族自治州',
        '红河州': '云南省红河哈尼族彝族自治州',
        '海西州': '青海省海西蒙古族藏族自治州',
        '清江浦': '江苏省淮安市清江浦区',
        '黔东南州': '贵州省黔东南苗族侗族自治州',
        '伊犁州': '新疆维吾尔自治区伊犁哈萨克自治州',
        '黔西南州': '贵州省黔西南布依族苗族自治州',
        '木渎镇': '江苏省苏州市吴中区木渎镇',
        '南沙海关': '广州市南沙区南沙海关',
        '建国门': '北京市东城区建国门',
        '台湖镇': '北京市通州区台湖镇',
        '阿坝州': '四川省阿坝藏族羌族自治州',
        '大柴旦': '青海省海西蒙古族藏族自治州大柴旦行政区',
        '丰台': '北京市丰台区',
        '天河': '广东省广州市天河区',
        '前门': '北京市东城区前门',
        '海淀': '北京市海淀区',
        '集同': '厦门市集同',
        '平谷': '北京市平谷区',
        '甘南州': '甘肃省甘南藏族自治州',
        '甪直镇': '江苏省苏州吴中区甪直镇',
        '穗东': '广州市黄埔区穗东',
        '增城': '广州市黄埔区增城海关',
        '江永': '湖南省永州市江永县',
        '海沧': '福建省厦门市海沧区',
        '可克达拉': '新疆维吾尔自治区可克达拉市',
        '青原': '江西省吉安市青原区',
        '交道口': '北京市东城区交道口',
        '东疆': '天津市东疆综合保税区'
    }
    
    s_auth = \
    df_event['处罚机关 Penalty Enforcement Authority'].replace(r'[A-Z\d]+', '', regex=True).replace(replace_dict, regex=True)
    
    # get region info
    df_region = extract_authorites(s_auth)

    # append to original df: province, city, area, level
    return df_event.join(df_region) 


# In[ ]:


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
    df_event = pd.read_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'), dtype={
        'start_year': int
    })
    
    print('add columns: region, law, fine amount, type...')
    
    df_event = df_event.pipe(add_penalty_law).pipe(add_penalty_type).pipe(add_fine).pipe(add_regions)
    
    # handle unknown provinces
    df_event = standard_province_names(df_event)

    print('check distributions...')
    
    draw_bar(df_event.start_year)

    draw_distribution_pie(df_event.province)

    draw_distribution_pie(df_event.level)
    
    draw_distribution_pie(df_event.penalty_type)

    print('save processed df_event...')
    
    df_event[['company_name', 
               'theme',
               'province',
               'level',
               'authority',
               'penalty_type',
               'penalty_law',
               'penalty_10K',
               'start_year',
               'start_date'
              ]].to_excel(os.path.join(processed_event_path, 'event1_penalty.xlsx'), 
                          index=False, 
                          freeze_panes=(1, 1))

    
    print('generate cross tables of campany and categories...')

    cat_column_names = ['level', 'theme', 'penalty_type']
    cat_prefix = ['E1_Penalty_Auth_level:', 'E1_Penalty_Theme:', 'E1_Penalty_Type:']
    number_column_name = 'penalty'
    multi_flags = [False, False, True]

    df_crosstab_event = get_crosstabs_for_multi_categories(
        df_event, 
        cat_prefix, 
        cat_column_names, 
        number_column_name, 
        multi_flags
    )
    
    # add fine sum
    s_fine_sum = df_event.groupby(['company_name'])['penalty_10K'].sum()

    df_crosstab_event = df_crosstab_event.reset_index(
                    ).set_index('company_name'
                   ).join(s_fine_sum).reset_index().set_index(['ID', 'company_name'])
    
    print('generate cross tables of [campany, year] and categories...')

    df_crosstab_event_year = get_crosstabs_for_multi_categories_by_year(
        df_event, 
        cat_prefix, 
        cat_column_names, 
        number_column_name,
        multi_flags,
        'start_year'
    )
    
    # handle fine amount per year per company
    df_fine_year = df_event.groupby(['company_name', 'start_year'], as_index=False)['penalty_10K'].sum().rename(
        columns={'start_year': 'year'})
    
    df_crosstab_event_year = df_crosstab_event_year.reset_index(
                    ).merge(df_fine_year, how='left', on=['company_name', 'year']
                    ).set_index(['ID', 'company_name', 'year'])
       
    print('generate period table of [ID, company, date] and number...')
    
    df_period = get_period_df(df_event, 'penalty', 'start_date')
    
    save_meta_events(df_crosstab_event, df_crosstab_event_year, df_period, 'event1_penalty')
    
    return df_crosstab_event, df_crosstab_event_year, df_period
    
    


# In[ ]:




