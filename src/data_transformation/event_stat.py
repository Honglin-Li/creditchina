#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
event_stat

The module provides statistical analysis on the credit dataset.

The output tables and figures are stored in data/stat directory. 
Parts of them are used in the thesis.
"""


# In[2]:


import math

# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *
from src.data_transformation.pipeline import *


# # Stat-related functions
# 
# ## stat: meta

# In[22]:


def generate_meta_stat(df_meta, save_name):
    """
    Create, save, and return stat tables of metadata.
    Based on the filtered df_meta, The stat can be based on all companies/mothers/daughters
    
    Parameters
    ----------
    df_meta : DataFrame
        metadata DataFrame.
    save_name : str
        The file name to save the stat data. The tables will be saved in directory data/stat.
    
    Returns
    -------
    df_province : DataFrame
        Contains stat data by province. The stat data here consists of number of firm and event, avg number
    df_province_year : DataFrame
        A DataFrame with columns={province, year, #company}.
    df_relation : DataFrame
        Contains columns: company_name, M_D.
    df_year_company : DataFrame
        Contains 
    """
    # 476 companies no date
    # handle date
    df_meta['foundation_year'] = df_meta['foundation_date'].str[:4]
    df_meta.loc[df_meta['foundation_year'] == '— —', 'foundation_year'] = s_year
    df_meta['foundation_year'] = df_meta['foundation_year'].fillna(s_year)
    df_meta.foundation_year = df_meta.foundation_year.astype(int)
    
    df_foundation_year = df_meta.groupby('foundation_year', as_index=False)['company_name'].count().rename(
        columns={'company_name': 'company'}
    ) # foundation_year, company(number)
    
    df_year_company = df_foundation_year.set_index(
        'foundation_year'
    ).cumsum().reset_index().rename(
        columns={
            'foundation_year': 'year'
        })
    
    df_relation = df_meta[['company_name', 'M_D']]
    
    # region
    df_province = df_meta.groupby(
        'province', 
        as_index=False
    )['company_name'].count().rename(
        columns={'company_name': 'company'}
    ).sort_values('company', ascending=False)
    
    # add label stat to province
    df_province['red'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'red'].sum())
    df_province['black'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'black'].sum())
    
    df_province['red_prop'] = (df_province.red / df_province.company).round(3)
    df_province['black_prop'] = (df_province.black / df_province.company).round(3)
    
    # add event stast to province
    print(df_meta.columns)
    df_province['permit'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'permit'].sum())
    df_province['penalty'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'penalty'].sum())
    df_province['redlist'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'redlist'].sum())
    df_province['blacklist'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'blacklist'].sum())
    df_province['watchlist'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'watchlist'].sum())
    df_province['commit'] = df_province.province.apply(lambda p: df_meta.loc[df_meta.province==p, 'commit'].sum())
    
    # event number/company number
    df_province['avg_permit'] = (df_province.permit / df_province.company).round(3)
    df_province['avg_penalty'] = (df_province.penalty / df_province.company).round(3)
    df_province['avg_redlist'] = (df_province.redlist / df_province.company).round(3)
    df_province['avg_blacklist'] = (df_province.blacklist / df_province.company).round(3)
    df_province['avg_watchlist'] = (df_province.watchlist / df_province.company).round(3)
    df_province['avg_commit'] = (df_province.commit / df_province.company).round(3)
    
    # event number/red or black company number
    df_province['avg_redlist_red'] = (df_province.redlist / df_province.red).round(3)
    df_province['avg_blacklist_black'] = (df_province.blacklist / df_province.black).round(3)
    
    df_level = df_meta.groupby('level', as_index=False)['company_name'].count().rename(
        columns={'company_name': 'company'}
    )
    
    # corporate_type
    #df_corporate_type = df_meta.groupby('corporate_type', as_index=False)['company_name'].count().rename(columns={'company_name': 'company'})
    
    # flags
    dict_flag = {
        'red': df_meta.red.sum(),
        'black': df_meta.black.sum(),
        'green': df_meta.green.sum(),
        'grey': df_meta.grey.sum()
    }
    
    s_flag = pd.Series(dict_flag)
    
    # event numbers
    dict_events = {
        'permit': df_meta.permit.sum(),
        'penalty': df_meta.penalty.sum(),
        'redlist': df_meta.redlist.sum(),
        'blacklist': df_meta.blacklist.sum(),
        'watchlist': df_meta.watchlist.sum(),
        'commit': df_meta.commit.sum()
    }
    
    s_event = pd.Series(dict_events)
    
    # handle province year active companies
    df_province_year = prepare_year_column(df_meta, 'province', 'foundation_year').groupby(
        ['province', 'year'], as_index=False).count()
    
    # save
    with pd.ExcelWriter(os.path.join(stat_path, save_name)) as writer:
        df_year_company.to_excel(writer, index=False, sheet_name='year_company')
        df_province.to_excel(writer, index=False, sheet_name='province')
        df_level.to_excel(writer, index=False, sheet_name='level')
        df_foundation_year.to_excel(writer, index=False, sheet_name='year')
        df_province_year.to_excel(writer, index=False, sheet_name='province_year')
        s_flag.to_frame().to_excel(writer, sheet_name='flags')
        s_event.to_frame().to_excel(writer, sheet_name='events')
    
    return df_province, df_province_year, df_relation, df_year_company
        


# ## stat: event
# 
# - year 
# - province
# - category
# - overview
# - draw

# In[4]:


def generate_year_stat(df_event, crosstab_year, number_col, df_year_firm, start_year = 'start_year'):
    """
    Get event stat table by year.
    
    Parameters
    ----------
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    crosstab_year : DataFrame
        The corresponding crosstab_year of the event.
    number_col : str
        The column name of the event name in crosstab_year, e.g.Redlist_Type:A-level taxpayer.
    df_year_firm : DataFrame
        The DataFrame shows how many active firms there are in each year.
    start_year : str
        The column name of start year in df.
    
    Returns
    -------
    DataFrame
        A DataFrame with index=year, 
        columns={#new record, # active record, #new company, #active company, 
        #avg_new_record, #avg_active_record, 
        #new_firm_coverage, #active_firm_coverage}
    """
    # get new records and company number every year from event DataFrame
    s_new_record = df_event[start_year].value_counts()

    s_new_company = df_event.groupby(start_year)['company_name'].nunique()

    # get active records and active company number for each year from crosstab_year DataFrame
    s_active_record = crosstab_year[number_col].reset_index().groupby('year')[number_col].sum()

    s_active_company = crosstab_year.index.to_frame(index=False).year.value_counts()

    df_year_stat = pd.DataFrame({'new_record': s_new_record, 
                  'active_record': s_active_record, 
                  'new_company': s_new_company, 
                  'active_company': s_active_company,
                  'active_firms': df_year_firm.set_index('year')['company']})
    
    # fill the empty of active firms
    df_year_stat.active_firms = df_year_stat.active_firms.ffill()
    
    # proportion
    df_year_stat['avg_new_record'] = (df_year_stat.new_record / df_year_stat.new_company).round(3)
    df_year_stat['ave_active_record'] = (df_year_stat.active_record / df_year_stat.active_company).round(3)
    
    # add frim coverage
    df_year_stat['new_firm_coverage'] = (df_year_stat.new_company / df_year_stat.active_firms).round(3)
    df_year_stat['active_firm_coverage'] = (df_year_stat.active_company / df_year_stat.active_firms).round(3)
    
    return df_year_stat[df_year_stat.index >= s_year]


# In[5]:


df_province_en = pd.read_excel(os.path.join(attri_path, 'regions.xlsx'), sheet_name='省')[['省份', 'Province']].rename(
    columns={
        '省份': 'province',
        'Province': 'province_en'
    })


# In[6]:


def generate_province_data(df_event, df_province):
    """
    Get province table for event.
    
    Parameters
    ----------
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    df_province : DataFrame
        Contains the province info in meta data.
    
    Returns
    -------
    DataFrame
        A DataFrame with index=province, columns={#record, #company, #avg_record, #company_in_province, #company_prop}
    """
    # check if 'province'
    if 'province' not in df_event.columns:
        print('The event has no province column')
        return None
    
    s_record = df_event['province'].value_counts()
    
    # unique company number
    s_company = df_event.groupby('province')['company_name'].nunique()
    
    df_province_stat = pd.DataFrame({'record': s_record, 
                  'company': s_company}).sort_values('record', ascending=False)
    
    df_province_stat['avg_record'] = (df_province_stat.record / df_province_stat.company).round(3)
    
    # !!!add company number in each province
    df_province_stat = df_province_stat.merge(
        df_province[['province', 'company']].rename(columns={'company': 'company_in_province'}),
        how='left',
        left_index=True,
        right_on='province'
    )
    
    df_province_stat['company_prop'] = (df_province_stat.company / df_province_stat.company_in_province).round(3)
    
    # add province en
    df_province_stat = df_province_stat.merge(df_province_en, how='left', on='province')
    
    return df_province_stat.set_index('province_en').sort_values('record', ascending=False)


# In[7]:


def generate_province_year_data(df_event, df_province_year, save_prefix, start_year='start_year', end_year=None):
    """
    Get province year table for event.
    
    Parameters
    ----------
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    df_province_year : DataFrame
        The table is used to add total company number per year each company.
    save_prefix : str
        The prefix for figure files.
    start_year : str, default 'start_year'
        The column name of start year column.
    end_year : str, optinal
        The column name of end year column.
        
    Returns
    -------
    DataFrame
        A DataFrame with columns={province, year, #active_record, #company, #total_company, #new_record, active_record_prop, new_record_prop}
    """
    # get most columns
    df = generate_cat_year_data(df_event, 'province', save_prefix+'_province', start_year, end_year)
    
    # add company number per year per province
    df_province_year['year'] = df_province_year['year'].astype(str)
    
    df = df.merge(df_province_year.rename(columns={'company_name': 'total_company'}), how='left', on=['province', 'year'])
    
    # add proportion
    df['active_company_prop'] = (df.active_company / df.total_company).round(3)
    
    # draw    
    print('Active company proportion variations')
    draw_cat_year_variation(df, 'province_en', 'active_company_prop', save_prefix + '_province_active_company_prop.png')
    draw_cat_year_variation(df, 'province_en', 'new_record', save_prefix + '_province_new_record.png')
    
    return df


# In[8]:


def draw_cat_year_variation(df, cat_col_name, col_name, save_name):
    """
    Draw and save figure of category variation over years.
    
    Parameters
    ----------
    df : DataFrame
        Event DataFrame thant contains only one type of event.
    cat_col_name : str
        The column name of category column.
    col_name : str
        The column name of data to show.
    save_name : str
        The figure file name to save.
    """
    # year float to int
    df['year'] = df['year'].astype(str)
    
    # calculate row numbers
    cats = df[cat_col_name].unique()
    
    fig_count = len(cats)
    
    ncols = 4
    nrows = math.ceil(fig_count/ncols)
    
    # drawing
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(4*ncols, 3*nrows))

    for i, c in enumerate(cats):
        data = df.loc[df[cat_col_name]==c, ['year', col_name]].set_index('year')[col_name]
        
        if len(c) > 45: # if too long to show
            c = c[:45]
            
        if nrows > 1:
            axes[int(i/ncols), i % ncols].plot(data)
            axes[int(i/ncols), i % ncols].set_title(c)
        else:
            axes[i % ncols].plot(data)
            axes[i % ncols].set_title(c)
        
    if nrows > 1:
        # delete empty subplots
        del_number = ncols-fig_count % ncols

        for i in range(del_number):
            fig.delaxes(axes[nrows-1, ncols-i-1])
        
    fig.tight_layout()
    
    plt.savefig(os.path.join(stat_path, save_name))
    
    #plt.show()


# In[9]:


def draw_cat_year_variation_in_one_plot(df, cat_col_name, col_name, save_name):
    """
    Draw and save figure of category variation over years.
    
    Parameters
    ----------
    df : DataFrame
        Event DataFrame thant contains only one type of event.
    cat_col_name : str
        The column name of category column.
    col_name : str
        The column name of data to show.
    save_name : str
        The figure file name to save.
    """
    # year float to int
    df['year'] = df['year'].astype(str)
    
    # calculate row numbers
    cats = df[cat_col_name].unique()
    
    ds = {}
    
    for i, c in enumerate(cats):
        #data = df.loc[df[cat_col_name]==c, ['year', col_name]].set_index('year')[col_name]
        
        #plt.plot(data, label=c, sharex=True)
        
        ds[c] = df.loc[df[cat_col_name]==c, ['year', col_name]].set_index('year')[col_name]
        
    df_source = pd.DataFrame(ds).sort_index()
    df_source = df_source.drop(index='2022')
    
    df_source.plot(legend=True)
    #plt.legend()
    
    plt.savefig(os.path.join(stat_path, save_name))
    
    #plt.show()


# In[10]:


def generate_cat_year_data(M_D, df_event, cat, save_prefix, start_year='start_year', end_year=None):
    """
    Get category year table for event.
    
    Parameters
    ----------
    M_D : str
        Can be M or D to distinguish the save name.
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    cat : str
        The column name of a given category column.
    save_prefix : str
        The prefix for figure files.
    start_year : str, default 'start_year'
        The column name of start year column.
    end_year : str, optinal
        The column name of end year column.
        
    Returns
    -------
    DataFrame
        A DataFrame with columns={cat, year, #active_record, #company, #new_record, active_record_prop, new_record_prop}
    """
    # check multiple cateogry
    df_explode = df_event.copy()
    
    if isinstance(df_explode.loc[0, cat], list):
        df_explode[cat] = df_explode[cat].apply(lambda x:[c.strip() for c in x]) # remove space

        df_explode = df_explode.explode(cat, ignore_index=True)
        
    df_year = prepare_year_column(df_explode, cat, start_year, end_year).groupby(
            [cat, 'year'], as_index=False)
    
    # active record
    df = df_year.count().rename(
        columns={'company_name': 'active_record'})
    
    # new record
    df_new_record = df_explode.groupby([cat, start_year], as_index=False)['company_name'].count().rename(
        columns={'company_name': 'new_record',
                 start_year: 'year'})

    df = df.merge(df_new_record, how='left', on=[cat, 'year'])
    
    # new company
    df_year_company = df_explode.groupby([cat, start_year], as_index=False)['company_name'].nunique().rename(
        columns={'company_name': 'year_company',
                 start_year: 'year'})

    df = df.merge(df_year_company, how='left', on=[cat, 'year'])
    
    # add unique company number with the event records
    df_active_company= df_year['company_name'].nunique().rename(
        columns={'company_name': 'active_company'})
    
    df = df.merge(df_active_company, how='left', on=[cat, 'year'])
    
    # add proportion
    df['avg_new_record'] = (df.new_record / df.year_company).round(3)
    df['avg_active_record'] = (df.active_record / df.active_company).round(3)
    
    # province en
    if cat == 'province':
        # add province en
        df = df.merge(df_province_en, how='left', on='province')
        
        cat = 'province_en'
    
    # draw
    save_prefix = save_prefix + '_' + cat
       
    print('New record/Year company variations over year')
    draw_cat_year_variation_in_one_plot(df, cat, 'avg_new_record', M_D + '_' + save_prefix + '_avg_new_record.png')
    
    print('Active record/Active company variations over year')
    draw_cat_year_variation_in_one_plot(df, cat, 'avg_active_record', M_D + '_' + save_prefix + '_avg_active_record.png')
    
    draw_cat_year_variation_in_one_plot(df, cat, 'new_record', M_D + '_' + save_prefix + '_new_record.png')
    
    draw_cat_year_variation_in_one_plot(df, cat, 'active_record', save_prefix + '_active_record.png')
    
    return df


# In[11]:


def generate_category_data(df_event, cat):
    """
    Generate stat for each cateogry in event.
    
    Parameters
    ----------
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    cat : str
        The column name of category column.
        
    Returns
    -------
    DataFrame
    """      
    # check empty
    print(f'NA {cat} records: {df_event[cat].isnull().sum()}')

    # handle multi-choice cateogry
    if '[' in df_event.loc[0, cat]:
        # str to list
        df_event[cat] = df_event[cat].str.replace(r'[\[\'\]]+', '', regex=True).str.split(',') # , space issue

    df_source_data = df_event[['company_name', cat]].copy()

    if isinstance(df_source_data.loc[0, cat], list):
        df_source_data[cat] = df_source_data[cat].apply(lambda x:[c.strip() for c in x]) # remove space

        df_source_data = df_source_data.explode(cat, ignore_index=True)

    # total records
    df_overview = df_source_data[cat].value_counts().reset_index().rename(
        columns={
            'index': cat,
            cat: 'record'})

    # add total company
    df_overview = df_overview.merge(df_source_data.groupby(cat, as_index=False)['company_name'].nunique().rename(
                  columns={'company_name': 'company'}),
                  how='left',
                  on=cat
                 )

    # add avg
    df_overview['avg_record'] = (df_overview.record / df_overview.company).round(3)

    return df_overview


# In[12]:


def generate_multiple_category_data(M_D, df_event, cat_list, save_prefix, start_year='start_year', end_year=None):
    """
    Generate and Save stat for all the cateogries in event.
    
    Parameters
    ----------
    M_D : str
        Can be M or D to distinguish the save name.
    df_event : DataFrame
        Event DataFrame thant contains only one type of event.
    cat_list : list of str
        Each item is a category column name.
    save_prefix : str
        Like 'penalty'.
    start_year : str, default 'start_year'
        The column name of start year column.
    end_year : str, optinal
        The column name of end year column.
        
    Returns
    -------
    list of dict
        Each dict: key is the sheet_name to save, value is the DataFrame to save.
    """   
    results = []
    
    for cat in cat_list:
        # generate {cat, #record, #company, # avg_record}
        df_cat_overview = generate_category_data(df_event, cat)
        
        # generate cat-year version        
        df_cat_year = generate_cat_year_data(M_D, df_event, cat, save_prefix, start_year, end_year)
            
        print(df_cat_overview.head())
        print(df_cat_year.head())
        
        results.append({'sheet_name': cat+'_overview', 'df': df_cat_overview})
        results.append({'sheet_name': cat+'_year', 'df': df_cat_year})
    
    return results


# In[13]:


def generate_event_overview(df_event):
    """
    Get overview stat of event: record number, company number, etc.
    
    Parameters
    ----------
    df_event : DataFrame
        
    Returns
    -------
    Series
    """        
    dict_stat = {
        'record': df_event.shape[0],
        'company': len(df_event.company_name.unique()),
        'column_number': df_event.shape[1],
        'columns': df_event.columns.tolist()
    }
    
    s_overview = pd.Series(dict_stat)
    
    return s_overview


# In[14]:


def generate_event_stat_separately(M_D, event, ct_year, event_type, cat_col_l, df_year_firm, start_year_col='start_year', end_year_col=None):
    """
    Generate and save all the stat tables for a event.
    
    Parameters
    ----------
    M_D : str
        Can be M or D to distinguish the save name.
    event : DataFrame
    ct_year : DataFrame
    event_type : str
        The value is one of 'permit', 'penalty', 'commit', 'redlist', 'blacklist', 'watchlist'.
    cat_col_l : list of str
        Each item is a category name, e.g. 'theme' and 'level'.
    df_year_firm : DataFrame
        Can be df_year_company_m or df_year_company_d
    start_year_col : str, default 'start_year'
    end_year_col : str, default 'end_year'
    """
    # overview
    s_overview = generate_event_overview(event)
    
    # year
    df_year_stat = generate_year_stat(event, ct_year, event_type, df_year_firm, start_year_col)
    
    # category year variation (saved inside the func)
    results = generate_multiple_category_data(M_D, event, cat_col_l, event_type, start_year_col, end_year_col)
    
    """ DELETE Province analysis part
    # province
    if event_type == 'redlist':
        # add province column from meta
        event = event.merge(df_meta[['company_name', 'province']], how='left', on='company_name')
        
    # province. redlist has no province column
    df_province_stat = generate_province_data(event, df_province)

    # province year variation
    df_province_year_stat = generate_province_year_data(
        event, df_province_year, event_type, start_year_col, end_year_col)
    """
    #save
    with pd.ExcelWriter(os.path.join(stat_path, 'stat_' + event_type + '_' + M_D + '.xlsx')) as writer:
        # save
        #df_province_stat.to_excel(writer, sheet_name='province')
        #df_province_year_stat.to_excel(writer, index=False, sheet_name='year_province')
        
        # save category stat
        for result in results:
            result['df'].to_excel(writer, index=False, sheet_name=result['sheet_name'])
            
        df_year_stat.to_excel(writer, sheet_name='year')
        s_overview.to_frame().to_excel(writer, sheet_name='overview')


# In[15]:


def generate_event_stat(df_year_company_m, df_year_company_d, event, ct_year, event_type, cat_col_l, start_year_col='start_year', end_year_col=None):
    """
    Generate and save all the stat tables for a event.
    
    Parameters
    ----------
    df_year_company_m : DataFrame
    df_year_company_d : DataFrame
    event : DataFrame
    ct_year : DataFrame
    event_type : str
        The value is one of 'permit', 'penalty', 'commit', 'redlist', 'blacklist', 'watchlist'.
    cat_col_l : list of str
        Each item is a category name, e.g. 'theme' and 'level'.
    start_year_col : str, default 'start_year'
    end_year_col : str, default 'end_year'
    """
    #split M D
    event_m = event[event.M_D != 'D'].reset_index(drop=True)
    event_d = event[event.M_D == 'D'].reset_index(drop=True)
    
    ct_year_m = ct_year[ct_year.M_D != 'D']
    ct_year_d = ct_year[ct_year.M_D == 'D']
    
    generate_event_stat_separately(
        'M', 
        event_m, 
        ct_year_m, 
        event_type, 
        cat_col_l, 
        df_year_company_m, 
        start_year_col, 
        end_year_col)
    
    generate_event_stat_separately(
        'D', 
        event_d, 
        ct_year_d, 
        event_type, 
        cat_col_l, 
        df_year_company_d, 
        start_year_col, 
        end_year_col)


# In[16]:


def flow_chart_data(df_meta):
    """
    Generate data for flow chart.
    
    Parameters
    ----------
    df_meta : DataFrame
        Metadata.
    """
    event_type = ['permit', 'penalty', 'watchlist', 'commit']

    redlist_company = (df_meta.redlist>0).sum()
    blacklist_company = (df_meta.blacklist>0).sum()

    for t in event_type:
        print('\n', t)
        event_company = (df_meta[t]>0).sum()
        red = ((df_meta.redlist>0) & (df_meta[t]>0)).sum()
        black = ((df_meta.blacklist>0) & (df_meta[t]>0)).sum()

        print('effect redlist...')
        print(red / redlist_company * 100)

        print('effect blacklist...')
        print(black / blacklist_company * 100)
    
    print('\n categorization...')
    print(redlist_company/ df_meta.red.sum() * 100)
    print(blacklist_company/ df_meta.black.sum() * 100)
    
    
#flow_chart_data(df_meta)


# In[ ]:





# In[ ]:




