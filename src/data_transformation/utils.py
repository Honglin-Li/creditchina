#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
utils

A Module to provide utility functions for the package.
"""


# In[1]:


# common packages
import pandas as pd
import numpy as np
import os
import re
from datetime import date
import json
from tqdm import tqdm
import cpca
import seaborn as sns
import matplotlib.pyplot as plt

# multiprocessing
from joblib import Parallel, delayed, parallel_backend
#from pandarallel import pandarallel

tqdm.pandas()
#pandarallel.initialize(progress_bar=True)


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.config import *


# In[ ]:


# Chinese
import matplotlib.font_manager as fm

# for notebook
#%matplotlib inline

font_path = os.path.join(data_path, 'simsun.ttc')
prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.sans-serif"]=["simsun"] 
plt.rcParams["axes.unicode_minus"]=False

# for multiprocessing
cpu_count = os.cpu_count()

# load when need
df_ids = None


# In[50]:


# install chinese province city area package
#!pip install cpca


# # Common functions

# In[ ]:


def parse_ymd(s):
    """
    Parse string of year-month-day to date type.
    
    Parameters
    ----------
    s : str
        The format is yyyy-mm-dd.
    
    Returns
    -------
    Date
    """
    year, month, day = s.split('-')
    return date(int(year), int(month), int(day))


# In[15]:


def load_meta():
    """
    Load metadata and metadata_year tables.
    """
    df_meta = pd.read_excel(os.path.join(processed_data_path, 'metadata.xlsx'))
    df_meta_year = pd.read_excel(os.path.join(processed_data_path, 'metadata_year.xlsx'), index_col=[0, 1, 2])
    
    return df_meta, df_meta_year


# In[1]:


def get_meta_event_paths(name):
    """
    Get paths of meta tables of a given event(identified by name).
    
    Parameters
    ----------
    name : str
        Part file name of 3 table, e.g. event4.
        
    Returns
    -------
    crosstab_path : str
    crosstab_year_path : str
    crosstab_day_path : str
    """
    # file name
    crosstab_name = 'crosstab_' + name + '.xlsx'
    crosstab_year_name = 'crosstab_year_' + name + '.xlsx'
    crosstab_day_name = 'crosstab_day_' + name + '.xlsx'
    
    # file path
    crosstab_path = os.path.join(processed_meta_event_path, crosstab_name)
    crosstab_year_path = os.path.join(processed_meta_event_path, crosstab_year_name)
    crosstab_day_path = os.path.join(processed_meta_event_path, crosstab_day_name)
    
    return crosstab_path, crosstab_year_path, crosstab_day_path


# In[2]:


def save_meta_events(crosstab, crosstab_year, crosstab_day, name):
    """
    Save two types of cross tables to local.
    
    Parameters
    ----------
    crosstab : DataFrame
        Cross table of company and categories.
    crosstab_year : DataFrame
        Cross table of [company, year] and categories which has MultiIndex.
    crosstab_day : DataFrame
        The DataFrame contains columns: ID, company_name, date, number.
    name : str
        Part file name of crosstabs.
    """
    crosstab_path, crosstab_year_path, crosstab_day_path = get_meta_event_paths(name)
    
    crosstab.to_excel(
        crosstab_path,
        freeze_panes=(1, 2))
    
    crosstab_year.to_excel(
        crosstab_year_path, 
        freeze_panes=(1, 3))
    
    crosstab_day.to_excel(
        crosstab_day_path, 
        freeze_panes=(1, 3))


# In[1]:


# pie
def draw_distribution_pie(s):
    """
    Show value_counts DataFrame and draw a pie chart.
    
    Parameters
    ----------
    s : Series
        The data source.
    """
    # consider list series
    if isinstance(s[0], list):
        s = s.explode()
    
    dist = s.value_counts()
    display(dist.to_frame())
    pie = plt.pie(dist, labels = dist.index, autopct='%.1f%%')
    plt.show()
    


# In[17]:


# bar
def draw_bar(s):
    """
    Show value_counts DataFrame and draw a bar chart.
    
    Parameters
    ----------
    s : Series
        The data source.
    """
    dist = s.value_counts().sort_index()
    display(dist.to_frame())
    b = plt.bar(dist.index, dist)
    plt.bar_label(b)
    plt.show()
    
    


# In[ ]:


pic_type_dict = {
    'pie': draw_distribution_pie,
    'bar': draw_bar
}

def draw_distribution(data_list, type_list):
    """
    Show value_counts DataFrame and draw a bar chart for each given data and type.
    
    Parameters
    ----------
    data_list : list of Series
    type_list : list of str
        The item value can be one of the keys of pic_type_dict
    """   
    print('check distributions...')
    
    for data, pic_type in zip(data_list, type_list):
        pic_type_dict[pic_type](data)


# In[18]:


def extract_law(x):
    """
    Extract law name from text(in penalty context, the text is penalty basis & type of illegal behavior).
    
    The extracting rules: 
    1. basis with《》, extract the law inside the marks;
    2. else if basis contains 第XX条, extract from start to the position;
    3. else keep the basis.
    
    Parameters
    ----------
    x : str
        The text contains the name of law.
    
    Returns
    -------
    str
        Law name.
    """
    x = str(x)
    if ('《' in x ) and ('》' in x):
        return x[x.find('《') : x.find('》') + 1]
    
    match = re.search(r'第\w*条', x)
    if match:
        return x[:match.start()]
    
    return x


# In[ ]:


def standard_province_names(df_event, province_col_name='province'):
    """
    Mainly change the provinces which are not in province list to NA.
    
    Parameters
    ----------
    df_event : DataFrame
    province_col_name : str, default 'province'
        The column name of province column.
        
    Returns
    -------
    DataFrame
    """
    eligible_provinces = pd.read_excel(os.path.join(attri_path, 'regions.xlsx'), sheet_name='province_short').province.tolist()
    
    unknown_provinces = list(set(df_event['province'])-set(eligible_provinces))
    unknow_index = df_event.loc[df_event[province_col_name].isin(unknown_provinces)].index
    
    if len(unknown_provinces) > 0:
        print('Unknown provinces, please check...')
        print(unknown_provinces)
        
        # !!!check if some provinces have no records
        check = set(eligible_provinces) - set(df_event[province_col_name])
        
        if len(check) > 0:
            print('those provinces have no records in the event')
            print(check)
            
        # change unknown province name to NA
        df_event.loc[unknow_index, province_col_name] = np.nan
    
    return df_event


# # authority related functions

# In[48]:


def extract_authorites(auth_s):
    """
    Extract pure authorites without region.
    
    Parameters
    ----------
    auth_s : array-like
    
    Returns
    -------
    DataFrame
        Contains columns: p, c, a, level, authority
    """
    df_region = extract_regions(auth_s).rename(
        columns = {'address': 'authority'}
        )
    
    return df_region


# # category related functions

# In[ ]:


def add_single_choice_category_from_keywords(df_event, cat_column_name, cat_source_column_name, category_dict):
    """
    Add single choice category column to a DataFrame.
    
    Parameters
    ----------
    df_event : DataFrame
    cat_column_name : str
        Name of the new category column.
    cat_source_column_name : str
        The column which contains the category info.
    category_dict : dict of {'search keywords': 'corresponding category'}
    
    Returns
    -------
    DataFrame
        df_event with category column.
    """  
    # extract keywords from reason type
    reason_pattern = '|'.join(category_dict.keys())
    
    df_event[cat_column_name] = df_event[cat_source_column_name].str.extract(f'({reason_pattern})')

    # check whether the dict cover all the rows
    print('rows that are not covered by mapping dict')
    display(df_event[df_event[cat_column_name].isna()])

    # fill the new category column
    df_event[cat_column_name] = df_event[cat_column_name].map(category_dict)
    
    return df_event


# In[ ]:


def add_multi_choice_category_from_keywords(df_event, cat_column_name, cat_source_column_name, category_dict):
    """
    Add multi-choice category column to a DataFrame.
    
    Note: if wanna check whether the dict cover all the rows, run single-choice version.
    
    Parameters
    ----------
    df_event : DataFrame
    cat_column_name : str
        Name of the new category column.
    cat_source_column_name : str
        The column which contains the category info.
    category_dict : dict of {'search keywords': 'corresponding category'}
    
    Returns
    -------
    DataFrame
        The category column contains list of multi-categories
    """  
    # extract keywords from reason type
    reason_pattern = '|'.join(category_dict.keys())
    
    df_event[cat_column_name] = df_event[cat_source_column_name].str.findall(f'({reason_pattern})')
    
    # de-duplicates, get key values
    df_event[cat_column_name] = df_event[cat_column_name].apply(lambda x: [category_dict[cat] for cat in set(x)])
    
    return df_event


# In[ ]:


def get_uniform_table(df_event, start_year_c, start_date_c, period, list_type, list_name='redlist_type'):
    """
    Ger uniformed table which contains columns: company_name, start_year, end_year, start_date, end_date, type.
    
    Parameters
    ----------
    df_event : DataFrame
    start_year_c : str
        The name of start year column.
    start_date_c : str
        The name of start date column.
    period : int
        The validation years, like 3.
        If it is set to a minus number, that's means no validation period, it is set to current end date(2022).
    list_type : str
        The value of the column list_name, like "Tax blacklist".
    list_name : str
        The column name of type, like "redlist_type" or "blacklist_type".
    
    Returns
    -------
    DataFrame
    """
    if df_event.shape[0] == 0:
        return pd.DataFrame(columns=['company_name', 'start_year', 'end_year', 'start_date', 'end_date', list_name])
    
    df_event = df_event[['company_name', start_year_c, start_date_c]].rename(columns={start_year_c: 'start_year',
                                                                        start_date_c: 'start_date'})
    
    if period > 0:
        df_event['end_year'] = df_event.start_year + period
        df_event['end_date'] = df_event.start_date.apply(lambda x: str(int(x[:4]) + period) + x[4:])
    else:
        df_event['end_year'] = e_year
        df_event['end_date'] = e_date
        
    df_event[list_name] = list_type
    
    return df_event


# In[49]:


def move_last_column_to_first(df):
    """
    Move the last column to the first column.
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    DataFrame
    """
    # order
    c = df.columns.tolist()
    
    column_order = c[-1:] + c[:-1]
    df = df[column_order]
    
    return df


# In[ ]:


def add_margins(df, margin_column_name):
    """
    Add total number column and move to the first column.
    
    Parameters
    ----------
    df : DataFrame
    margin_column_name : str
    
    Returns
    -------
    DataFrame
    """
    df[margin_column_name] = df.sum(axis=1)
    
    df = move_last_column_to_first(df)
    
    return df


# In[ ]:


def assign_id_to_company(df):
    """
    Add ID to DataFrame.
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    DataFrame
    """
    global df_ids
    
    if not isinstance(df_ids, pd.DataFrame):
        df_ids = pd.read_excel(os.path.join(attri_path, 'company_ids.xlsx'))
    
    return df.merge(df_ids, how='left', on='company_name')


# # day-level event-number-table related functions

# In[ ]:


def get_period_df(df_event, no_col, start_date_col='start_date', end_date_col=None):
    """
    Get a DataFrame of event numbers of all the date change points of companies.
    
    Parameters
    ----------
    df_event : DataFrame
        The DataFrame to work on.
    no_col : str
        The name of new column which records the event(record) number of corresponding date change point of a company.
    start_date_col : str
        The name of start date column in df_event.
    end_date_col : str, optional
        The name of end date column in df_event.
        If null, fill end date from config (now: 2022-12-31)
    
    Returns
    -------
    DataFrame
        The DataFrame contains columns: ID, company_name, date, no_col.
    """
    period_l = [] # ds for the final DF
    
    # end_date column
    if end_date_col is None:
        end_date_col = 'end_date'
        df_event[end_date_col] = e_date
        
    # end_date may be some nan
    df_event[end_date_col] = df_event[end_date_col].fillna(e_date)

    # columns needed    
    df_period = df_event[[
        'company_name', 
        start_date_col, 
        end_date_col]]

    # out-loop: each company
    for firm_name, group in df_period.groupby('company_name'):
        # create a dict: key-date, value-delta(add/minus)
        date_dict = {}

        # inner-loop1: each record -> fill out date_dict
        for index, row in group.iterrows():
            # for start date, delta+1
            key = row[start_date_col]
            date_dict[key] = date_dict.get(key, 0) + 1

            # for end date, delta-1
            key = row[end_date_col]
            date_dict[key] = date_dict.get(key, 0) - 1

        # inner-loop2: each date point -> append final data row
        current_number = 0

        for date in sorted(date_dict):
            if date != e_date: #TODO: check logic, if real expire date is really equal to e_date
                current_number += date_dict[date]

            period_l.append({
                'company_name': firm_name,
                'date': date,
                no_col: current_number  
            })

    # to frame, add ID
    if len(period_l) == 0:
        return pd.DataFrame(columns=['ID', 'company_name', 'date', no_col])
    else:
        return pd.DataFrame(period_l).pipe(assign_id_to_company).set_index(['ID', 'company_name', 'date'])
    


# In[1]:


def concat_periods(period_l, margin_col_name=None):
    """
    Get a merged period table with margins and ffill in firm groups.
    
    Parameters
    ----------
    period_l : list of DataFrame
        Each item come from get_period_df().
    margin_col_name : str
        The name of new margin column.
    
    Returns
    -------
    DataFrame
    """
    # eliminate empty df
    period_l = [p for p in period_l if p.shape[0]>0]
    
    # concat all the period tables
    df_period = pd.concat(period_l, join='outer', axis=1).reset_index().sort_values(by=['ID', 'date'], ignore_index=True)

    # ffill by company, then add margin
    df_period = df_period.groupby('ID').fillna(method='ffill').\
        pipe(assign_id_to_company).\
        set_index(['ID', 'company_name', 'date'])
    
    if margin_col_name:
        df_period = df_period.pipe(add_margins, margin_col_name)
    
    return df_period


# # frequency table related functions

# In[20]:


def prepare_year_column(df_event, cat_col_name, start_year, end_year=None):
    """
    Add and explode column: year between start years and end years.
    
    Parameters
    ----------
    df_event : DataFrame
        The DataFrame to work on.
    cat_col_name : str
        The categorical column name in df_event.
    start_year : str
        The start year column name in df_event.
    end_year : str, optional
        The end year column name in df_event.
        If null, fill end year from config (now: 2022)
    
    Returns
    -------
    DataFrame
        The processed df_event with 3 columns: company_name, category, year.
    """
    if end_year is None:
        end_year = 'end_year'
        df_event[end_year] = e_year
    
    # df: the columns we need    
    df = df_event[['company_name', cat_col_name, start_year, end_year]]
    df = df.astype({start_year: 'int',
               end_year: 'int'})
    
    # reset years
    df[start_year] = df[start_year].apply(lambda x: max(x, s_year))
    df[end_year] = df[end_year].apply(lambda x: min(x, e_year))
    
    # check end_year>start_year, then delete
    tmp = df[end_year] - df[start_year]
    del_id = tmp[tmp < 0].index
    df.drop(del_id, inplace=True)
    
    # get year list then explode
    df['year'] = df.apply(lambda x: list(range(x[start_year], x[end_year]+1)), axis=1)
    df = df.explode('year', ignore_index=True).drop([start_year, end_year], axis=1)
    
    # may exist duplicate years after explode
    #df.drop_duplicates(inplace=True)
    
    return df



# In[46]:


def get_crosstab(df, prefix, cat_col_name, multi_flag=False):
    """
    Get cross table for company and single-choice & multi-choice categories.
    
    For multi-choice categories, explode the category column first.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to work on.
    prefix : str
        Prefix to new category columns.
    cat_col_name : str
        The categorical column name in df_event.
    multi_flag : bool, default False
        Flag to show the category is single choice or multi-choice.
    
    Returns
    -------
    DataFrame
        The cross table with ID.
    """        
    if multi_flag:
        df = df.explode(cat_col_name, ignore_index=True)
        
    df_freq = pd.crosstab(index=df['company_name'], 
                          columns=df[cat_col_name]
                         ).add_prefix(prefix).reset_index().pipe(assign_id_to_company).set_index(['ID', 'company_name'])
    
    return df_freq


# In[47]:


def get_crosstab_by_year(df, prefix, cat_col_name, start_year, end_year=None, multi_flag=False):
    """
    Get cross table for [company, year] and categories.
    
    For multi-choice categories, explode the category column first.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to work on.
    prefix : str
        Prefix to new category columns.
    cat_col_name : str
        The categorical column name in df_event.
    start_year : str
        The start year column name in df_event.
    end_year : str, optional
        The end year column name in df_event.
    multi_flag : bool, default False
        Flag to show the category is single choice or multi-choice.
    
    Returns
    -------
    DataFrame
        The cross table with MultiIndex.
    """    
    if multi_flag:
        df = df.explode(cat_col_name, ignore_index=True)
        
    df = prepare_year_column(df, cat_col_name, start_year, end_year)
    # now, the df contains 3 columns: company_name, cat_columne, year, run crosstab to get frequency table
            
    return pd.crosstab(
        index=[df['company_name'], df['year']], 
        columns=df[cat_col_name]
    ).add_prefix(prefix).reset_index().pipe(assign_id_to_company).set_index([
        'ID',
        'company_name',
        'year'])


# In[30]:


def get_crosstabs_for_multi_categories(df_event, cat_prefix, cat_column_names, number_column_name, multi_flags):
    """
    Get cross tables for company and multiple categories.
    
    Better be sure the first category is a single-choice category, so that the total number is corresponding to real record number.
    
    Parameters
    ----------
    df_event : DataFrame
        The DataFrame to work on.
    cat_prefix : list of str
        The corresponding prefix to new category columns.
    cat_column_names : list of str
        The corresponding categorical column name in df_event.
    number_column_name : str
        The name of total number column.    
    multi_flags : list of bool
        Indicate the category column is multi-choice or not.
        
    Returns
    -------
    DataFrame
        The cross table for multiple categories which contains company_name, ID, #number, category columns.
    """
    dfs_freq = []
    
    for c, p, f in zip(cat_column_names, cat_prefix, multi_flags):
        dfs_freq.append(
            get_crosstab(df_event, p, c, f)
        )
    
    # add total numbers to the first df_freq
    dfs_freq[0] = add_margins(dfs_freq[0], number_column_name)

    
    return pd.concat(dfs_freq, axis=1)


# In[27]:


def get_crosstabs_for_multi_categories_by_year(df_event, cat_prefix, cat_column_names, number_column_name, multi_flags, start_year, end_year=None):
    """
    Get cross tables for [company, year] and multiple categories.
    
    Parameters
    ----------
    df_event : DataFrame
        The DataFrame to work on.
    cat_prefix : list of str
        The corresponding prefix to new category columns.
    cat_column_names : list of str
        The corresponding categorical column name in df_event.
    number_column_name : str
        The name of total number column.
    multi_flags : list of bool
        Indicate the category column is multi-choice or not.
    start_year : str
        The start year column name in df_event.
    end_year : str, optional
        The end year column name in df_event.
    
    Returns
    -------
    DataFrame
        The cross table for multiple categories which contains MultiIndex(company_name, ID, year), #number, category columns.
    """    
    if end_year is None:
        end_year = 'end_year'
        df_event[end_year] = e_year
        
    dfs_freq = []

    # get freq_df by year for each category
    for c, p, f in zip(cat_column_names, cat_prefix, multi_flags):
        tmp = get_crosstab_by_year(
            df_event, 
            p, 
            c, 
            start_year, 
            end_year,
            f
        )

        dfs_freq.append(tmp)

    # add total numbers    
    dfs_freq[0] = add_margins(dfs_freq[0], number_column_name)
    
    # -------for df_meta_year_eventX: details
    df_freq_year = pd.concat(dfs_freq, axis=1)
    
    return df_freq_year


# In[ ]:


def create_all_crosstabs(
    df_event,
    cat_prefix, 
    cat_column_names, 
    number_column_name, 
    multi_flags,
    save_prefix,
    start_year_col,
    start_date_col,
    end_year_col=None,
    end_date_col=None
):
    """
    Genarate, Save and Return crosstab, crosstab_year, crosstab_period for one given event.
    
    Parameters
    ----------
    df_event : DataFrame
        The DataFrame to work on.
    cat_prefix : list of str
        The corresponding prefix to new category columns.
    cat_column_names : list of str
        The corresponding categorical column name in df_event.
    number_column_name : str
        The name of total number column.
    multi_flags : list of bool
        Indicate the category column is multi-choice or not.
    save_prefix : str
        The prefix of file name of 3 crosstabs.
    start_year_col : str
        The start year column name in df_event.
    end_year_col : str, optional
        The end year column name in df_event.
    start_date_col : str
        The start year column name in df_event.
    end_date_col : str, optional
        The end year column name in df_event.
    
    Returns
    -------
    df_crosstab_event : DataFrame
    df_crosstab_event_year : DataFrame
    df_period : DataFrame
    """  
    print('generate cross tables of campany and categories...')

    df_crosstab_event = get_crosstabs_for_multi_categories(
        df_event, 
        cat_prefix, 
        cat_column_names, 
        number_column_name, 
        multi_flags
    )
    
    print('generate cross tables of [campany, year] and categories...')

    df_crosstab_event_year = get_crosstabs_for_multi_categories_by_year(
        df_event, 
        cat_prefix, 
        cat_column_names, 
        number_column_name,
        multi_flags,
        start_year_col,
        end_year_col
    )
    
    print('generate period table of [ID, company, date] and number...')
    
    df_period = get_period_df(df_event, number_column_name, start_date_col, end_date_col)
    
    save_meta_events(df_crosstab_event, df_crosstab_event_year, df_period, save_prefix)
    
    return df_crosstab_event, df_crosstab_event_year, df_period


# # region related functions
# 
# extract region, level ,province, city, area from authroties and regions.xlsx

# In[28]:


def assign_region_level(row): # from province, county, city columns
    """
    Assign region level(province, city, area) to the row.
    
    This func is used in :meth:`pandas.DataFrame.apply`.
    
    Parameters
    ----------
    row : array-like
    
    Returns
    -------
    str
        Return the region level
    """
    if pd.notnull(row['area']):
        return 'area'
    if pd.notnull(row['city']):
        return 'city'
    if pd.notnull(row['province']):
        return 'province'

    return 'uncertain'


def fill_pca(row, pca, address_pattern): # for NA
    """
    Fill pca(province, city, area) for rows.
    
    This func is used in :meth:`pandas.DataFrame.apply`.
    
    Parameters
    ----------
    row : array-like
        DataFrame row.
    pca : DataFrame
        A DataFrame which contains pca to full pca info.
    address_pattern : regex pattern
        A regex pattern to split a full pca to (p, c, a)
    
    Returns
    -------
    tuple of (p, c, a)
    """
    if pd.notnull(row['省']):
        # some areas are not accounted into area, but in address
        if bool(re.match(r'.+区', row['地址'])) & pd.isnull(row['区']):
            row['区'] = row['地址']
            
        return row['省'], row['市'], row['区']
    
    # dealing with NA: find area-> find corresponding pca in pca -> extract p, c, a from pca
    keyword = row['a']

    if pd.isnull(row['a']) & pd.notnull(row['c']):
        keyword = row['c']
    
    full_pca = pca.loc[pca['region'] == keyword, 'pca'].tolist()
    if len(full_pca) == 1:
        full_pca = full_pca[0]
        return address_pattern.match(full_pca).groups() # (p, c, a)
    else:
        return row['p'], row['c'], row['a']
    
    
def extract_regions(region_s):
    """
    Extract region related columns from an address-like column.
    
    The function firstly use cpca package to extract region info. Then for those who could not be extracted, extract by regex.
    
    Parameters
    ----------
    region_s : array-like
        A column of DataFrame which contains address-like values, can be authority names in our case.
    
    Returns
    -------
    DataFrame
        A DataFrame which contains columns: province, city, area, level.
        If a region column is needed, it can be attained by concat p, c, a
    """
    # get pca from package
    df_region = cpca.transform(region_s)[['省', '市', '区', '地址']]
    
    # split region to p, c , a
    address_pattern = re.compile(r'(?P<p>[^省]+自治区|.*?省|上海市|北京市|天津市|重庆市)?(?P<c>[^市]+自治州|.*?地区|.+盟|.*?市)?(?P<a>[^县]+县|.+区|.+市|.+旗)?')
    df_region = df_region.join(region_s.str.extract(address_pattern, expand=True))
    
    # deal with those which can not be transformed
    pca = pd.read_excel(os.path.join(attri_path, 'regions.xlsx'), sheet_name='pca') # province city area
    df_pca = df_region.apply(fill_pca, args=(pca, address_pattern), axis=1, result_type='expand')
    df_pca.columns = ['province', 'city', 'area']
    
    # assign level
    df_pca['level'] = df_pca.apply(assign_region_level, axis=1)
    df_pca = df_pca.join(df_region['地址']).rename(
        columns={'地址': 'address'})
    
    return df_pca # province| city| area| level | address


# In[ ]:




