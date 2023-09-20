#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
event2_processing

Processing event2 redlist.

Contents:
    INPUT:
    - clean_sub_events/custom, a_taxpayer, highway, transportation


    OUTPUT:
    - processed_data/events/event2.xlsx
        - company_name
        - start_year, start_date
        - end_year, end_date
        - redlist_type: 4 types of redlists
    - processed_data/meta_events/crosstab_event2.xlsx
    - processed_data/meta_events/crosstab_year_event2.xlsx
        - company_name, ID, year, #redlist, #custom, #a_taxpayer, #highway, #transportation
    - processed_data/meta_events/crosstab_day_event2.xlsx
        - ID, company_name, date, redlist, redlist_custom, redlist_tax, redlist_highway, redlist_transportation


    PROCESS:
    - combine 4 sub_event tables  into one table
    - for end year, date: 
        - from 2014, advanced certificated companies: is 3 years
        - a level taxpayer: 1 year
        - highway & transportation: 2 years
    - generate 3 tables in meta_events direcotry
    - custom have 2 problems:
        - duplicates(by company_name, regester date, without level date)
        - missing data in level year, fill out by regester year
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# In[2]:


def load_data():
    df_event_custom = pd.read_excel(os.path.join(sub_event_path, '21_event_custom.xlsx'))
    df_event_a_taxpayer = pd.read_excel(os.path.join(sub_event_path, '22_event_a_taxpayer.xlsx'))
    df_event_highway = pd.read_excel(os.path.join(sub_event_path, '23_event_highway.xlsx'))
    df_event_transportation = pd.read_excel(os.path.join(sub_event_path, '24_event_transportation.xlsx'))
    
    return df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation


# In[3]:


# clean custom data
def get_uniform_table_for_custom(df_event_custom):
    # deal with duplicates
    c = df_event_custom.duplicated(['company_name', 'register_year']).sum()
    print(f'{c} rows have the same company names and register years')
    
    # add duplicate count
    count_s = df_event_custom.company_name.value_counts()
    df_event_custom['count'] = df_event_custom.company_name.map(count_s)
    
    # delete the rows which count=2 & level_year = nan
    del_id = df_event_custom[(df_event_custom['count']==2) & (df_event_custom.level_year.isna())].index
    df_event_custom.drop(del_id, inplace=True)
    
    # fill missing level years
    nan_id = df_event_custom[df_event_custom.level_year.isna()].index
    df_event_custom.loc[nan_id, 'level_year'] = df_event_custom.loc[nan_id, 'register_year']
    df_event_custom.loc[nan_id, 'level_date'] = df_event_custom.loc[nan_id, 'register_date']
    
    # integrate into redlist event
    df_event_custom = get_uniform_table(df_event_custom, 'level_year', 'level_date', 3, 
                                     'Custom advanced enterprice')
    
    return df_event_custom


# In[4]:


def get_uniform_table_for_taxpayer(df_event_a_taxpayer):
    # check missing data
    c = df_event_a_taxpayer['评价年度 Evaluation year'].isna().sum()
    print(f'{c} missing evaluation years')
    
    # integrate into redlist
    df_event_a_taxpayer['start_year'] = df_event_a_taxpayer['评价年度 Evaluation year'] + 1
    df_event_a_taxpayer['start_date'] = df_event_a_taxpayer.start_year.apply(
        lambda x: str(x) + '-06-01') # TODO: check date
    df_event_a_taxpayer = get_uniform_table(df_event_a_taxpayer, 'start_year', 'start_date', 1, 
                                     'A-level taxpayer')
    
    return df_event_a_taxpayer


# In[5]:


def process_event(use_validation=False):
    """
    Get crosstabs(& year) for blacklists by processing 5 redlist-related sub events.
    
    Parameters
    ----------
    use_validation : bool, default False
        if False -> end_year=2022, do not consider the validation years.
    
    Returns
    -------
    df_crosstab_redlists : DataFrame
    df_crosstab_year_redlists : DataFrame
    df_period : DataFrame
    """
    # load data
    df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation = load_data()

    # ---------------get uniform tables for redlists---------------
    # custom
    print('generate table for custom...')
    df_event_custom = get_uniform_table_for_custom(df_event_custom)

    # distribution
    print('distribution of custom by level years')
    
    draw_bar(df_event_custom.start_year)

    # taxpayer
    print('generate table for A-level taxpayer...')
    df_event_a_taxpayer = get_uniform_table_for_taxpayer(df_event_a_taxpayer)
    
    print('distribution of a-taxpayer by evaluation years')
    print(df_event_a_taxpayer.shape)
    draw_bar(df_event_a_taxpayer.start_year)

    # highway
    print('generate table for Trustworthy enterprises of highway construction...')
    # add start date
    df_event_highway['start_date'] = '2020-07-16'
    df_event_highway.loc[df_event_highway['年度 Year'] == 2018, 'start_date'] = '2018-05-15'
    
    df_event_highway = get_uniform_table(df_event_highway, '年度 Year', 'start_date', 2, 
                                         'Trustworthy enterprises of highway construction')
    
    print('distribution of highway by evaluation years')
    print(df_event_highway.shape)
    draw_bar(df_event_highway.start_year)

    # transportation
    print('generate table for Trustworthy enterprises of transportation construction...')
    df_event_transportation['start_year'] = 2020
    df_event_transportation['start_date'] = '2020-07-16'
    df_event_transportation = get_uniform_table(df_event_transportation, 
                                                'start_year', 'start_date', 2, 
                                                'Trustworthy enterprises of transportation construction')
    
    print(df_event_transportation.shape)

    # ---------------concat all the redlists to one and save---------------
    print('connect the 4 tables...')
    df_event_redlists = pd.concat([df_event_custom, df_event_a_taxpayer, df_event_highway, df_event_transportation])
    df_event_redlists.to_excel(os.path.join(processed_event_path, 'event2.xlsx'), index=False, freeze_panes=(1,1))
    
    # ---------------period---------------
    print('period table...')
    
    # by document rule or system rule
    end_year = None
    end_date = None
    if use_validation:
        end_year = 'end_year'
        end_date = 'end_date'
    
    # individual period tables
    df_period_custom = get_period_df(df_event_custom, 'redlist_custom', 'start_date', end_date)
    df_period_tax = get_period_df(df_event_a_taxpayer, 'redlist_tax', 'start_date', end_date)
    df_period_highway = get_period_df(df_event_highway, 'redlist_highway', 'start_date', end_date)
    df_period_transportation = get_period_df(df_event_transportation, 'redlist_transportation', 'start_date', end_date)
    
    # combine to one
    df_period = concat_periods([df_period_custom, df_period_tax, df_period_highway, df_period_transportation], 'redlist')
    
    display(df_period.head())
    
    # ---------------stat---------------
    print('crosstab...')
    
    # crosstab
    df_crosstab_redlists = get_crosstab(df_event_redlists, 'Redlists:', 'redlist_type')

    # add total number
    df_crosstab_redlists = add_margins(df_crosstab_redlists, 'redlist')
    display(df_crosstab_redlists.head())
    
    # ---------------stat by year---------------
    print('crosstab year...')        
    df_crosstab_year_redlists = get_crosstab_by_year(
        df_event_redlists, 'Redlist_Type:', 'redlist_type', 'start_year', end_year)

    df_crosstab_year_redlists = add_margins(df_crosstab_year_redlists, 'redlist')

    display(df_crosstab_year_redlists.head())
    
    
    # save
    save_meta_events(df_crosstab_redlists, df_crosstab_year_redlists, df_period, 'event2')

    return df_crosstab_redlists, df_crosstab_year_redlists, df_period


# df_crosstab_redlists, df_crosstab_year_redlists, df_period = process_event()

# In[ ]:




