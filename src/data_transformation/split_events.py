#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
split_events

Split and pre-processing metadata and event data.

Contents:
    INPUT
    - data/original/mothers_original.xlsx
    - data/original/daughters_original.xlsx

    OUTPUT
    - data/original/all_companies_original.xlsx
    - data/clean_sub_events/metadata&xxx_sub_events.xlsx

    PROCESS
    - remove space around company_name
    - merge mothers and daughters into one data source
    - split 5 events to sub-events
    - do some basis preprocessing for each split
        - remove duplicates
        - extract years
        - fill missing values
        - clean authority columns and fill out wrong ones with data sources
        - unify np.nan, like ——， /， 无
    - extract #sub_events to Metadata, add ID, re-order
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *
from src.data_transformation.add_themes import add_penalty_themes, add_permit_themes, add_commitment_themes
import calendar


# # 1. Merge Daughters and Mothers
# - merge, drop the extra info columns in daughters(they are from those companies whose USCI start from none-9)
# 

# In[2]:


def remove_space_in_name(df, name = 'company_name'):
    """
    Remove the space after company_name column.
    
    Parameters
    ----------
    df : DataFrame
    name : str
        The column name containing company names
    """
    df[name] = df[name].str.strip()

def combine_str_columns(row):
    """
    Combine the text in all the cells in a row.
    
    Parameters
    ----------
    row : Series
    
    Returns
    -------
    str
    """
    return ''.join(row.dropna().astype(str))


# In[3]:


def merge_companies():
    """
    Merge mothers and daughters by meta and individual events, then save to local.
    
    Before merge, handle daughters metadata: remove space around company names, merge representative person column, 
    fill out empty address with approval authorities, merge firm types.
    
    Returns
    -------
    df_meta : DataFrame
        Merged meta DataFrame.
    df_event1 : DataFrame
        Merged event1 DataFrame.
    df_event2 : DataFrame
        Merged event2 DataFrame.
    df_event3 : DataFrame
        Merged event3 DataFrame.
    df_event4 : DataFrame
        Merged event4 DataFrame.
    df_event5 : DataFrame
        Merged event5 DataFrame.
    """
    # if there is merged data, just return
    merged_path = os.path.join(original_path, 'all_companies_original.xlsx')
    
    if os.path.exists(merged_path):
        with pd.ExcelFile(merged_path) as xls:
            df_meta = pd.read_excel(xls, sheet_name='metadata')
            df_event1 = pd.read_excel(xls, sheet_name='event1')
            df_event2 = pd.read_excel(xls, sheet_name='event2')
            df_event3 = pd.read_excel(xls, sheet_name='event3')
            df_event4 = pd.read_excel(xls, sheet_name='event4')
            df_event5 = pd.read_excel(xls, sheet_name='event5')
            
        return df_meta, df_event1, df_event2, df_event3, df_event4, df_event5
    
    # load mothers and daughters
    with pd.ExcelFile(os.path.join(original_path, 'mothers_original.xlsx')) as xls:
        df_meta_m = pd.read_excel(xls, sheet_name='metadata')
        df_event1_m = pd.read_excel(xls, sheet_name='event1')
        df_event2_m = pd.read_excel(xls, sheet_name='event2')
        df_event3_m = pd.read_excel(xls, sheet_name='event3')
        df_event4_m = pd.read_excel(xls, sheet_name='event4')
        df_event5_m = pd.read_excel(xls, sheet_name='event5')

    with pd.ExcelFile(os.path.join(original_path, 'daughters_original.xlsx')) as xls:
        df_meta_d = pd.read_excel(xls, sheet_name='metadata')
        df_event1_d = pd.read_excel(xls, sheet_name='event1')
        df_event2_d = pd.read_excel(xls, sheet_name='event2')
        df_event3_d = pd.read_excel(xls, sheet_name='event3')
        df_event4_d = pd.read_excel(xls, sheet_name='event4')
        df_event5_d = pd.read_excel(xls, sheet_name='event5')
    
    # handle daughters meta
    # address
    
    df_meta_d['住所 Address'] = df_meta_d[['住所 Address', '审批机关 Approving authority']].apply(combine_str_columns, axis=1)
    #df_meta_d['住所 Address'].fillna('').astype(str) + \
        #df_meta_d['审批机关 Approving authority'].fillna('').astype(str)
    
    # representative
    df_meta_d['法定代表/负责人/执行事务合伙人 Legal representative/Person in charge/Executive business partner'] = \
        df_meta_d[['法定代表/负责人/执行事务合伙人 Legal representative/Person in charge/Executive business partner', 
            '法定代表人姓名 Name of Legal representative',
            '法定代表人 Legal representative'
           ]].apply(combine_str_columns, axis=1)
       # df_meta_d['法定代表/负责人/执行事务合伙人 Legal representative/Person in charge/Executive business partner'].fillna('').astype(str) + \
       # df_meta_d['法定代表人姓名 Name of Legal representative'].fillna('').astype(str) + \
       # df_meta_d['法定代表人 Legal representative'].fillna('').astype(str)
    
    # firm type
    df_meta_d['企业类型 Corporate type'] = df_meta_d[['企业类型 Corporate type', 
                                              '组织类型 Organization type'
                                          ]].apply(combine_str_columns, axis=1)
    #df_meta_d['企业类型 Corporate type'].fillna('').astype(str) + \
       # df_meta_d['组织类型 Organization type'].fillna('').astype(str)
    
    # drop the extra columns in metadata of daughters, then merge
    df_meta = pd.concat([df_meta_m, 
                        df_meta_d[['统一社会信用代码 Unified Social Credit Identifier, USCI',
                                   '机构名称 Institution name', 
                                   '企业类型 Corporate type', 
                                   '住所 Address',
                                   '法定代表/负责人/执行事务合伙人 Legal representative/Person in charge/Executive business partner',
                                   '成立日期 Date of Foundation', 
                                   '行政管理 Administrative management',
                                   '严重失信主体名单 Severe untrustworthy entity list', 
                                   '信用承诺 Credit commitment',
                                   '司法判决 Judicial decision', 
                                   '诚实守信 Honesty and trustworthy',
                                   '经营异常 Abnormal operation', 
                                   '信用评价 Credit assessment', 
                                   '其他信息 Other information', 
                                   'black',
                                   'green', 
                                   'red', 
                                   'grey',
                                   'event_count', 
                                   'n_pages',
                                   'M_D']]])
    df_meta.grey = df_meta.grey.fillna(0)
    
    # remove space in company_name
    remove_space_in_name(df_meta, '机构名称 Institution name')

    df_event1 = pd.concat([df_event1_m, df_event1_d])
    remove_space_in_name(df_event1)
    
    df_event2 = pd.concat([df_event2_m, df_event2_d])
    remove_space_in_name(df_event2)
    
    df_event3 = pd.concat([df_event3_m, df_event3_d])
    remove_space_in_name(df_event3)
    
    df_event4 = pd.concat([df_event4_m, df_event4_d])
    remove_space_in_name(df_event4)
    
    df_event5 = pd.concat([df_event5_m, df_event5_d])
    remove_space_in_name(df_event5)
    
    # save the merged version
    with pd.ExcelWriter(merged_path) as xls:
        df_meta.to_excel(xls, sheet_name='metadata', index=False, freeze_panes=(1, 2))
        df_event1.to_excel(xls, sheet_name='event1', index=False, freeze_panes=(1, 2))
        df_event2.to_excel(xls, sheet_name='event2', index=False, freeze_panes=(1, 2))
        df_event3.to_excel(xls, sheet_name='event3', index=False, freeze_panes=(1, 2))
        df_event4.to_excel(xls, sheet_name='event4', index=False, freeze_panes=(1, 2))
        df_event5.to_excel(xls, sheet_name='event5', index=False, freeze_panes=(1, 2))
    
    return df_meta, df_event1, df_event2, df_event3, df_event4, df_event5


# # 2. USCI
# - USCI: extract organizational type. because few companies not start from 9(business)
# - region-USCI (Some codes do not match the 2020 area codes）
# 
# ## 2.1 extract organizational type

# In[4]:


def extract_organ_types(df_meta):
    """
    Add organ_type column from USCI to df_meta.
    
    Parameters
    ----------
    df_meta : DataFrame
    
    Returns
    -------
    DataFrame
        The df_meta with a new column: organ_type.
    """
    # load organizational type
    region_path = os.path.join(attri_path, 'regions.xlsx')
    df_organ_type = pd.read_excel(region_path, sheet_name='USCI', dtype={'code12': str})[['code12', 'type', 'type_en']]

    # look into organizational type
    organ_type_stat = df_meta[usci].str[:2].value_counts().to_frame().reset_index().rename(
            columns={'index': 'code12',
                    usci: 'count'})
    organ_type_stat.merge(df_organ_type, how='left')

    # from the result above, we found 2 codes without info, so add code12=22 manually
    df_organ_type = pd.concat([df_organ_type, pd.DataFrame([
        {'code12': '22', 'type': '外交', 'type_en': 'Diplomatic'},
        {'code12': '——', 'type': '没有USCI', 'type_en': 'no USCI'}
    ])], ignore_index=True)

    display(df_organ_type.tail())

    # add organizational type to metadata
    organ_type_dict = dict(zip(df_organ_type['code12'], df_organ_type['type_en']))
    df_meta['organ_type'] = df_meta[usci].str[:2].replace(organ_type_dict)
    return df_meta


# ## 2.2 extract Region: level, region, province

# In[5]:


def add_regions(df_meta):
    """
    Extract and add level, region, province columns to df_meta from USCI.
    
    If there is no corresponding region info in the USCI, extract region info from address column.
    
    Parameters
    ----------
    df_meta : DataFrame
    
    Returns
    -------
    DataFrame
        The df_meta with 3 region columns.
    """
    def combine_data_sources(df_meta, df_pca):
        na_id = df_meta[df_meta.province.isnull()].index
        df_meta.loc[na_id, 'level'] = df_pca.loc[na_id, 'level']
        df_meta.loc[na_id, 'province'] = df_pca.loc[na_id, 'province']
        df_meta.loc[na_id, 'region'] = df_pca.loc[na_id, 'region']
        
    def change_individual_region(df_meta, wrong_province_value, l, p, r):
        na_id = df_meta[df_meta.province == wrong_province_value].index
        df_meta.loc[na_id, 'level'] = l
        df_meta.loc[na_id, 'province'] = p
        df_meta.loc[na_id, 'region'] = r
        
    # load region: code, level, region, province
    region_path = os.path.join(attri_path, 'regions.xlsx')
    df_region = pd.read_excel(region_path, dtype={'code': str}, sheet_name='CODE6')
    
    # DS1: add region from usci
    df_meta['code'] = df_meta[usci].str[2:8]
    df_meta = df_meta.merge(df_region, how='left', on='code')
    
    # check how many companies have no corresponding region
    print(f'{df_meta.level.isnull().sum()} companies have no corresponding region info from their USCI, need to extract from address')
    
    # DS2: extract region info from address
    df_pca = extract_regions(df_meta['住所 Address']) 
    df_pca['region'] = df_pca[['province', 'city', 'area']].apply(combine_str_columns, axis=1) # province| city| area| level | address | region
    
    # combine the 2 data source
    combine_data_sources(df_meta, df_pca)
    
    # manually change 2 errors in region
    change_individual_region(df_meta, '洪泽经济开发区328省', 'area', '江苏省', '江苏省淮安市洪泽区')
    change_individual_region(df_meta, '京山市雁门口镇(107省', 'area', '湖北省', '湖北省荆门市京山市雁门口镇')
    
    # DS3: for those without province, use company_name search for pca
    df_pca = extract_regions(df_meta[company_name]) 
    df_pca['region'] = df_pca[['province', 'city', 'area']].apply(combine_str_columns, axis=1)
    
    # combine the 2 data source
    combine_data_sources(df_meta, df_pca)
    
    
    # drop code6
    df_meta.drop(['code'], axis=1, inplace=True)

    return df_meta
    


# # 3. Split events to sub-events
# - split
# - preprocessing: fill missing values, drop duplicates, remove useless columns, merge events

# ### common functions

# In[6]:


def check_equal(df_event, bool_list):
    """
    Check if the sub-events are splited correctly.
    
    Parameters
    ----------
    df_event : DataFrame
    bool_list : list of bool Series
        The item is a bool Series which mark each row is belong to a sub-event or not.
    """
    sub_sum = sum([l.sum() for l in bool_list])
    total = df_event.shape[0]
    print(f'the number of sub events:{sub_sum}')
    print(f'total number: {total}')
    
    equal = total - sub_sum
    if equal == 0:
        print('Correct')
    else:
        print('Incorrect, please check the numbers of sub-events')
        for l in bool_list:
            print(l.sum())
            


# In[7]:


def check_duplicates(df_events):
    """
    Check duplicates of events.
    
    Parameters
    ----------
    df_events : list of DataFrame
        Each DataFrame is for a sub-event.
    
    Returns
    -------
    list of int
        Each item is the number of duplicates of the sub-event.
    """
    results = []
    print('duplicates:')
    
    for df in df_events:
        result = df.duplicated().sum()
        print(result)
        results.append(result)
        
    return results


# In[8]:


def extract_year(df, date_column_name, prefix, fill_value=np.nan):
    """
    Extract year from a date column & rename the date column.
    
    Parameters
    ----------
    df : DataFrame
        A DataFrame containing a date column.
    date_column_name : str
        The name of the data column.
    prefix : str
        The prefix of date and year columns, like "start_" or "end_".
    fill_value : str or int, default np.nan
        If the date is nan, fill NA by fill_value.
    """
    # if date is not a date, fill a value
    df[prefix + 'year'] = df[date_column_name].str[:4].replace({'— —': fill_value})
    
    return df.rename(columns={
        date_column_name : prefix + 'date'
    })


# In[9]:


def has_numbers(text):
    """
    Return if a text contains numbers(>=2).
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    bool
    """
    return bool(re.search(r'\d{2,}', str(text)))


# In[10]:


def extract_start_year(df_event, source_column):
    """
    Extract start date and year columns from content column for those events without start dates.
    
    Parameters
    ----------
    df_event : DataFrame
    source_column : str
        The column name which contains dates.
    
    Returns
    -------
    DataFrame
        A DataFrame with start year and missing flag columns.
    """
    date_pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})?日?'
    
    # func to set start date
    def get_date(x): # x: list of tuples, may like (2019, 1, 31) or (2019, 1, )
        if len(x):
            x = x[-1] # get the last date tuple
            
            # add last day
            if not x[2]:
                last_day_of_month = calendar.monthrange(int(x[0]), int(x[1]))[1]
                x = list(x)
                x[2] = str(last_day_of_month)
                
            return '-'.join(x)
        
        return np.nan
        
    
    df_event['start_date'] = df_event[source_column].str.findall(date_pattern).map(get_date)
    
    # fill NA with MAX year
    if df_event.start_date.isnull().sum():
        # get max date
        max_date = pd.to_datetime(df_event.start_date).max().strftime('%Y-%m-%d')
        df_event['flag'] = np.nan

        df_event.loc[df_event.start_date.isnull(), 'flag'] = 'Missing date'
        df_event.start_date = df_event.start_date.fillna(max_date)
    
    # add year
    df_event['start_year'] = df_event.start_date.str[:4]
    #extract_year(df_event, 'start_date', 'start_')
    
    return df_event


# ## 2.1 event1

# In[11]:


def split_event1(df_event1):
    """
    Split event1 into permit and penalty.
    
    The processing contains extract year, fill incorrect authorites by data sources, fill na, drop dupliacates.
    
    Parameters
    ----------
    df_event1 : DataFrame
    
    Returns
    -------
    df_event_permit : DataFrame
    df_event_penalty : DataFrame
    """
    # filter condition to get sub-events
    bool_event_permit_old = df_event1['审核类型 Audit Type'].notna()
    bool_event_permit_new = df_event1['许可类别 Type of Permission'].notna()
    bool_event_penalty = df_event1['处罚机关 Penalty Enforcement Authority'].notna()
    
    # check the total amount
    check_equal(df_event1, [bool_event_permit_old, bool_event_permit_new, bool_event_penalty])
    # difference is 37. the 37 entries are from non-business companies, so can be ignored
    
    # get the df for all the sub-events
    df_event_permit_old = df_event1.loc[bool_event_permit_old, ['统一社会信用代码 Unified Social Credit Identifier', 
                                       'company_name',
                                       '行政许可决定文书号 Administrative Permission Decision Document Code', 
                                       '许可决定日期 Permission decision date', 
                                       '许可内容 Permission Content', 
                                       '许可机关 Permission Authority', 
                                       '许可截止日期 Permission deadline date', 
                                       '审核类型 Audit Type']].reset_index(drop=True)

    df_event_permit_new = df_event1.loc[bool_event_permit_new, [
        '统一社会信用代码 Unified Social Credit Identifier', 
        'company_name',
        '行政许可决定文书号 Administrative Permission Decision Document Code',
        '行政许可决定文书名称 Name of Administrative Permission Decision',
        '许可证书名称 Name of Permission Certificate', 
        '许可类别 Type of Permission',
        '许可编号 Number of Permission', 
        '许可决定日期 Permission decision date',
        '有效期自 Permission  Decision Valid From', 
        '有效期至 Valid Until',
        '许可内容 Permission Content', 
        '许可机关 Permission Authority',
        '许可机关统一社会信用代码 Permission Authority USCI', 
        '数据来源单位 Data Sources Unit',
        '数据来源单位统一社会信用代码 Data Sources Unit USCI']].reset_index(drop=True)

    df_event_penalty = df_event1.loc[bool_event_penalty, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '行政处罚决定书文号 Administrative Penalty Decision Document Number',
            '处罚类别 Penalty Type', 
            '处罚决定日期 Penalty Date', 
            '处罚内容 Penalty Content',
            '罚款金额(万元) Fine Amount, 10k Yuan',
            '没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan',
            '暂扣或吊销证照名称及编号 Suspension or Revocation of License Name and Number',
            '违法行为类型 Type of Illegal Behavior', 
            '违法事实 Illegal Facts',
            '处罚依据 Penalty Basis', 
            '处罚机关 Penalty Enforcement Authority',
            '处罚机关统一社会信用代码 Penalty Enforcement Authority USCI', 
            '数据来源 Data sources',
            '数据来源单位统一社会信用代码 Data Sources Unit USCI']].reset_index(drop=True)
    
    # check duplicates
    results = check_duplicates([df_event_permit_old, df_event_permit_new, df_event_penalty])
    
    # preprocessing: drop duplicates
    if results[0] > 0:
        df_event_permit_old.drop_duplicates(inplace=True)
    if results[1] > 0:
        df_event_permit_new.drop_duplicates(inplace=True)
    if results[2] > 0:
        df_event_penalty.drop_duplicates(inplace=True)
    
    # preprocessing: permit old
    df_event_permit_old = df_event_permit_old.pipe(
        extract_year, 
        '许可决定日期 Permission decision date', 
        'start_'
    ).pipe(extract_year, 
           '许可截止日期 Permission deadline date', 
           'end_', 
           '2099')
    
    # permit new
    # extract year
    df_event_permit_new = df_event_permit_new.pipe(
        extract_year, 
        '许可决定日期 Permission decision date', 
        'start_'
    ).pipe(extract_year, 
           '有效期至 Valid Until', 
           'end_', 
           '2099')
    
    # fill out missing permit authorities by data source
    # if authority has numbers, means a wrong value
    # authorities is mess, like 浙江省行政权力运行系统-2, 2022-09-31, 东环函〔2022〕13号, 39f58a70-a952-4ab2-bf3f-f1aea2655584, 闽侯县城乡规划局(2), 汕头市自然资源局(根据省政府令第270号,受省林业局委托), 信阳市-信阳市产业集聚区-信阳市工业城工五路10号

    tmp = df_event_permit_new['许可机关 Permission Authority'].apply(has_numbers)
    wrong_auth_index = tmp[tmp].index
    print('those authorities will be filled out by data source')
    display(df_event_permit_new.loc[wrong_auth_index, ['许可机关 Permission Authority', '数据来源单位 Data Sources Unit']].head())
    
    # fill wrong auth
    df_event_permit_new.loc[wrong_auth_index, '许可机关 Permission Authority'] = df_event_permit_new.loc[wrong_auth_index, '数据来源单位 Data Sources Unit']
    df_event_permit_new.loc[wrong_auth_index, '许可机关统一社会信用代码 Permission Authority USCI'] = df_event_permit_new.loc[wrong_auth_index, '数据来源单位统一社会信用代码 Data Sources Unit USCI']
    
    # delete useless columns
    df_event_permit_new.drop(['有效期自 Permission  Decision Valid From', '许可编号 Number of Permission', '数据来源单位 Data Sources Unit', '数据来源单位统一社会信用代码 Data Sources Unit USCI'], axis=1, inplace=True)
    
    # combine permit old and new version
    df_event_permit = pd.concat([df_event_permit_new, df_event_permit_old.rename(
        columns={'审核类型 Audit Type': '许可类别 Type of Permission',
                '许可截止日期 Permission deadline date': '有效期至 Valid Until'})])
    df_event_permit['permit_type'] = df_event_permit['许可类别 Type of Permission'].str[:2] # remove the text after type"others"
    
    # PENALTY
    # year
    df_event_penalty = df_event_penalty.pipe(
        extract_year, 
        '处罚决定日期 Penalty Date', 
        'start_'
    )
    
    # penalty auth
    # some wrong values with date, remove date
    def clean_text(text):
        return re.sub(r'(\s处罚日期:)?\d{4}年\d{1,2}月\d{1,2}日', '', text)


    df_event_penalty['处罚机关 Penalty Enforcement Authority'] = df_event_penalty['处罚机关 Penalty Enforcement Authority'].apply(
        clean_text)

    # numbers
    tmp = df_event_penalty['处罚机关 Penalty Enforcement Authority'].apply(has_numbers)
    wrong_auth_index = tmp[tmp].index
    print('those authorities will be filled out by data sources')
    display(df_event_penalty.loc[wrong_auth_index, ['处罚机关 Penalty Enforcement Authority', '数据来源 Data sources']])
    
    df_event_penalty.loc[wrong_auth_index, '处罚机关 Penalty Enforcement Authority'] = df_event_penalty.loc[wrong_auth_index, '数据来源 Data sources']
    df_event_penalty.loc[wrong_auth_index, '处罚机关统一社会信用代码 Penalty Enforcement Authority USCI'] = df_event_penalty.loc[wrong_auth_index, '数据来源单位统一社会信用代码 Data Sources Unit USCI']
    
    # change money missing values to 0
    df_event_penalty['罚款金额(万元) Fine Amount, 10k Yuan'] = df_event_penalty['罚款金额(万元) Fine Amount, 10k Yuan'].replace({
        '— —': 0
    })
    df_event_penalty['没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan'] = df_event_penalty['没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan'].replace({
        '— —': 0
    })
    
    # remove values with null meaning to null
    df_event_penalty['暂扣或吊销证照名称及编号 Suspension or Revocation of License Name and Number'] = df_event_penalty[
        '暂扣或吊销证照名称及编号 Suspension or Revocation of License Name and Number'].replace(
        {'— —': np.nan,
        '无': np.nan,
        '/': np.nan})
    
    df_event_penalty.drop(['数据来源 Data sources', '数据来源单位统一社会信用代码 Data Sources Unit USCI'], axis=1, inplace=True)
    
    # Add authority theme and save permit and penalty
    df_event_permit = add_permit_themes(df_event_permit)
    #df_event_penalty = add_penalty_themes(df_event_penalty)
    
    # save the 4 sub-events
    print('saving(take time)...')
    df_event_permit_old.to_excel(os.path.join(sub_event_path, '13_event_permit_old.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_permit_new.to_excel(os.path.join(sub_event_path, '14_event_permit_new.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_permit.to_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_penalty.to_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'), index=False, freeze_panes=(1, 2))
          
    return df_event_permit, df_event_penalty


# ## 2.2 Split event2

# In[12]:


def split_event2(df_event2):
    """
    Split event2 into 4 types of redlists.
    
    Parameters
    ----------
    df_event2 : DataFrame
    
    Returns
    -------
    df_event_custom : DataFrame
    df_event_a_taxpayer : DataFrame
    df_event_highway : DataFrame
    df_event_transportation : DataFrame
    """
    # row condition
    bool_event_custom = df_event2['海关注册编码 Customs Record Number'].notna()
    bool_event_a_taxpayer = df_event2['纳税人名称 Taxpayer name'].notna()
    bool_event_highway = df_event2['企业资质 Firm Qualification'].notna()
    bool_event_transportation = df_event2['文件依据 Document basis'].notna()
    
    check_equal(df_event2, [bool_event_custom, bool_event_a_taxpayer, bool_event_highway, bool_event_transportation])
    
    # split
    df_event_custom = df_event2.loc[bool_event_custom, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '组织机构代码 Organizational institution identifier',
            '海关注册编码 Customs Record Number', 
            '首次注册日期 First registration date',
            '等级认定时间 Accreditation time',
            '数据来源 Data sources']].reset_index(drop=True)

    df_event_a_taxpayer = df_event2.loc[bool_event_a_taxpayer, [
        '统一社会信用代码 Unified Social Credit Identifier', 
        'company_name',
        '评价年度 Evaluation year',
        '数据来源 Data sources']].reset_index(drop=True) # taxpayer and id repeat

    df_event_highway = df_event2.loc[bool_event_highway, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '注册地址 Registration address', 
            '企业资质 Firm Qualification', 
            '年度 Year',
            '备注 Comment',
            '数据来源 Data sources']].reset_index(drop=True)

    df_event_transportation = df_event2.loc[bool_event_transportation, [
        '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '文件依据 Document basis',
            '数据来源 Data sources']].reset_index(drop=True)
    
    # duplicates
    results = check_duplicates([df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation])
    
    df_event_a_taxpayer.drop_duplicates(inplace=True)
    
    # custom
    df_event_custom = df_event_custom.pipe(
        extract_year, 
        '首次注册日期 First registration date', 
        'register_'
    ).pipe(
    extract_year,
    '等级认定时间 Accreditation time', 
    'level_')
    
    # save
    print('saving...')
    df_event_custom.to_excel(os.path.join(sub_event_path, '21_event_custom.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_a_taxpayer.to_excel(os.path.join(sub_event_path, '22_event_a_taxpayer.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_highway.to_excel(os.path.join(sub_event_path, '23_event_highway.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_transportation.to_excel(os.path.join(sub_event_path, '24_event_transportation.xlsx'), index=False, freeze_panes=(1, 2))
    
    return df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation
    


# ## 2.3 split event3
# 

# In[18]:


def split_event3(df_event3):
    """
    Split event3 into 5 types of blacklists.
    
    Parameters
    ----------
    df_event3 : DataFrame
    
    Returns
    -------
    df_event_dishonest_person : DataFrame
    df_event_safety_production : DataFrame
    df_event_tax_blacklist : DataFrame
    df_event_gov_procure_illegal : DataFrame
    df_event_overload_transport_illegal : DataFrame
    """
    bool_event_dishonest_person = df_event3['省份  Province'].notna()
    bool_event_safety_production = df_event3['纳入理由 Reason for inclusion'].notna()
    bool_event_tax_blacklist = df_event3['纳税人识别号 Taxpayer identification number'].notna()
    bool_event_gov_procure_illegal = df_event3['处罚截止日期 Penalty deadline'].notna()
    bool_event_overload_transport_illegal = df_event3['入库时间 Registration time'].notna()
    check_equal(df_event3, [bool_event_dishonest_person, bool_event_safety_production, bool_event_tax_blacklist, bool_event_gov_procure_illegal, bool_event_overload_transport_illegal])
    
    # split
    df_event_dishonest_person = df_event3.loc[bool_event_dishonest_person, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '执行法院 Execution Court',
            '省份  Province', 
            '执行依据文号 Execution Base Number', 
            '立案时间 Date of Filing',
            '案号 Case Number', 
            '做出执行依据单位 Execution Base Unit',
            '生效法律文书确定的义务 Obligations Determined by Effective Legal Documents',
            '被执行人的履行情况 Implementation Performance of Person of Execution',
            '失信被执行人行为具体情形 Person of Execution Untrustworthy Behavior Details',
            '发布时间 Date of Issue', 
            '已履行部分 Part of Accomplishment',
            '未履行部分 Part of Non-accomplishment', 
            '数据来源 Data sources'
    ]].reset_index(drop=True)
    
    # change province(short version) in system to province in standard version
    df_province_trans = pd.read_excel(os.path.join(attri_path, 'regions.xlsx'), sheet_name='province_short', index_col=0)
    
    df_event_dishonest_person['省份  Province'] = df_event_dishonest_person['省份  Province'].map(
        df_province_trans.province.to_dict()
    ) 

    df_event_safety_production = df_event3.loc[bool_event_safety_production, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '主要负责人 Principal', 
            '注册地址 Registration address',
            '失信行为简况 Default behavior profile',
            '信息报送机关 Information reporting authority', 
            '纳入理由 Reason for inclusion',
            '数据来源 Data sources'
    ]].reset_index(drop=True)

    df_event_tax_blacklist = df_event3.loc[bool_event_tax_blacklist, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '纳税人名称 Taxpayer name',
            '法定代表人或负责人姓名 Name of legal representative or person in charge',
            '纳税人识别号 Taxpayer identification number', 
            '案件上报期 Case reporting period',
            '负有直接责任的财务负责人姓名 Name of the financial person directly responsible',
            '负有直接责任的中介机构信息及其从业人员信息 Information of intermediaries with direct responsibility and information of their practitioners',
            '案件性质 Nature of the case',
            '主要违法事实 Main illegal facts',
            '相关法律依据及税务处理处罚 Relevant legal basis and tax treatment and punishment',
            '注册地址 Registration address',
            '数据来源 Data sources'
    ]].reset_index(drop=True)

    df_event_gov_procure_illegal = df_event3.loc[bool_event_gov_procure_illegal, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '企业地址 Business address',
            '不良行为的具体情形 Specific circumstances of the bad behavior',
            '处罚依据 Penalty basis', '处罚结果 Penalty result', 
            '记录日期 Record date',
            '登记地点 Registration location', 
            '处罚截止日期 Penalty deadline',
            '数据来源 Data sources'
    ]].reset_index(drop=True)

    df_event_overload_transport_illegal = df_event3.loc[bool_event_overload_transport_illegal, [
            '统一社会信用代码 Unified Social Credit Identifier', 
            'company_name',
            '入库时间 Registration time',
            '车牌号 License number',
            '营运证号 Operation certificate number', '道路运输证号 Road transport certificate number', 
            '批次 Batch',
            '失信行为 Breach of trust',
            '数据来源 Data sources'
    ]].reset_index(drop=True)
    
    # year
    
    # dishonest person
    df_event_dishonest_person = df_event_dishonest_person.pipe(
        extract_year, '立案时间 Date of Filing', 'case_').pipe(
        extract_year, '发布时间 Date of Issue', 'issue_')
    
    # gov
    if df_event_gov_procure_illegal.shape[0] > 0:
        df_event_gov_procure_illegal = df_event_gov_procure_illegal.pipe(
            extract_year, '记录日期 Record date', 'start_').pipe(
            extract_year, '处罚截止日期 Penalty deadline', 'end_')
    
    if df_event_overload_transport_illegal.shape[0] > 0:
        df_event_overload_transport_illegal = df_event_overload_transport_illegal.pipe(
            extract_year, '入库时间 Registration time', 'start_')
    
    if df_event_tax_blacklist.shape[0] > 0:
        df_event_tax_blacklist = df_event_tax_blacklist.pipe(
            extract_start_year, '主要违法事实 Main illegal facts')
    
    if df_event_safety_production.shape[0] > 0:
        df_event_safety_production = df_event_safety_production.pipe(
            extract_start_year, '失信行为简况 Default behavior profile')

    print('saving...')
    df_event_dishonest_person.to_excel(os.path.join(sub_event_path, '31_event_dishonest_person.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_safety_production.to_excel(os.path.join(sub_event_path, '32_event_safety_production.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_tax_blacklist.to_excel(os.path.join(sub_event_path, '33_event_tax_blacklist.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_gov_procure_illegal.to_excel(os.path.join(sub_event_path, '34_event_gov_procure_illegal.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_overload_transport_illegal.to_excel(os.path.join(sub_event_path, '35_event_overload_transport_illegal.xlsx'), index=False, freeze_panes=(1, 2))

    return df_event_dishonest_person, df_event_safety_production, df_event_tax_blacklist, df_event_gov_procure_illegal, df_event_overload_transport_illegal


# ## 2.4 event4 clean

# In[19]:


def split_event4(df_event4):
    """
    Clean event4.
    
    Parameters
    ----------
    df_event4 : DataFrame
    
    Returns
    -------
    DataFrame
    """
    df_event4 = df_event4.pipe(
        extract_year, '设定日期 Establishment date', 'start_')
    
    print('saving...')
    df_event4[['统一社会信用代码 Unified Social Credit Identifier', 'company_name',
           '列入经营异常名录原因类型名称 Reason type for listing in Operational abnormality',
           'start_date', '列入决定机关名称 Listing decision authority name',
           '数据来源 Data sources', 'start_year']].to_excel(os.path.join(sub_event_path, '4_event_abnormal_operations.xlsx'), index=False, freeze_panes=(1, 2))
    
    return df_event4


# ## 2.5 Split event5

# In[23]:


def split_event5(df_event5):
    """
    Split event5 into 2 types of credit commitments.
    
    Parameters
    ----------
    df_event5 : DataFrame
    
    Returns
    -------
    df_event_implementation : DataFrame
    df_event_public : DataFrame
    """
    bool_event_implementation = df_event5['承诺类型 Commitment type'].notna()

    bool_event_public = df_event5['经办人 Manager'].notna()

    check_equal(df_event5, [bool_event_implementation, bool_event_public])
    
    # split
    df_event_implementation = df_event5.loc[bool_event_implementation, [
        '统一社会信用代码 Unified Social Credit Identifier', 'company_name',
        '承诺类型 Commitment type', '承诺事由 Commitment reason',
           '做出承诺日期 Commitment date', '承诺受理单位 Commitment processing unit'
    ]].reset_index(drop=True)

    df_event_public = df_event5.loc[bool_event_public, [
        '统一社会信用代码 Unified Social Credit Identifier', 'company_name',
        '信用承诺事项 Credit Commitment Matters',
        '做出信用承诺时间 Time to make a credit commitment'
    ]].reset_index(drop=True)
    
    # duplicates
    results = check_duplicates([df_event_implementation, df_event_public])
    
    df_event_implementation.drop_duplicates(inplace=True)
    df_event_public.drop_duplicates(inplace=True)
    
    df_event_implementation = df_event_implementation.pipe(
        extract_year, '做出承诺日期 Commitment date', 'start_')
    df_event_public = df_event_public.pipe(
        extract_year, '做出信用承诺时间 Time to make a credit commitment', 'start_')
    
    # add theme
    #df_event_implementation = add_commitment_themes(df_event_implementation)
    
    print('saving...')
    df_event_implementation.to_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'), index=False, freeze_panes=(1, 2))
    df_event_public.to_excel(os.path.join(sub_event_path, '52_event_commitment_public.xlsx'), index=False, freeze_panes=(1, 2))
    
    return df_event_implementation, df_event_public


# ## final function to run all the functions above

# In[24]:


def split_events():
    
    """
    Split and pre-process events and metadata(organ types & region info).
    """   
    df_meta, df_event1, df_event2, df_event3, df_event4, df_event5 = merge_companies()

    df_meta = df_meta.pipe(extract_organ_types).pipe(add_regions)

    # TODO:mannully fill out the 40 NA region info, since there is no way to identify the region info.(the meta in processed_data)

    # delete 0 number columns 
    df_meta.drop(['司法判决 Judicial decision', '信用评价 Credit assessment', '其他信息 Other information'], axis=1, inplace=True)

    # check distributions
    draw_distribution_pie(df_meta.organ_type)
    draw_distribution_pie(df_meta.province)
    draw_distribution_pie(df_meta.level)

    print('split event1...')
    df_event_permit, df_event_penalty = split_event1(df_event1)

    print('split event2...')
    df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation = split_event2(df_event2)

    # add lastest evaluation year of A taxpayer
    df_year = df_event_a_taxpayer.groupby('统一社会信用代码 Unified Social Credit Identifier', as_index=False).max('评价年度 Evaluation year').rename(columns={
            '统一社会信用代码 Unified Social Credit Identifier': '统一社会信用代码 Unified Social Credit Identifier, USCI',
            '评价年度 Evaluation year': 'Latest evaluation year for A taxpayer'})
    df_meta = df_meta.merge(df_year, how='left')

    print('split event3...')
    df_event_dishonest_person, df_event_safety_production, df_event_tax_blacklist, \
    df_event_gov_procure_illegal, df_event_overload_transport_illegal = split_event3(df_event3)

    print('clean event4...')
    df_event4 = split_event4(df_event4)

    print('split event5...')
    df_event_implementation, df_event_public = split_event5(df_event5)

    # add id
    df_meta = df_meta.reset_index().rename(columns={'index': 'ID'})
    
    # save a clean version for data transformation, leave the minimum columns
    df_meta['foundation_year'] = df_meta['成立日期 Date of Foundation'].str[:4]
    
    df_meta['foundation_year'] = df_meta['foundation_year'].fillna(s_year) # TODO: 476 companies without foundation year.
    
    print('save meta...')
    df_meta.to_excel(os.path.join(sub_event_path, 'metadata.xlsx'), index=False, freeze_panes=(1, 3))
    
    df_meta = df_meta[['ID', 
                 '机构名称 Institution name',
                 '企业类型 Corporate type',
                 'level', 'region', 'province', 'M_D', 'organ_type',
                 'foundation_year',
                 '成立日期 Date of Foundation',
                 'Latest evaluation year for A taxpayer',
                 'green', 'red', 'black', 'grey',
                ]].rename(columns={'机构名称 Institution name': 'company_name',
                                  '成立日期 Date of Foundation': 'foundation_date',
                                  '企业类型 Corporate type': 'corporate_type'})

    print('save clean meta...')
    df_meta.to_excel(os.path.join(processed_data_path, 'metadata.xlsx'), index=False, freeze_panes=(1, 2))
    
    # save a excel with only company_name and ID
    print('save ID-company...')
    df_meta[['ID', 'company_name']].to_excel(os.path.join(attri_path, 'company_ids.xlsx'), index=False)


# split_events()

# In[ ]:




