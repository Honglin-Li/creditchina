#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
event3_processing

Processing event3 blacklists.

Contents:
    INPUT:
    ---
    - clean_sub_events/event3 related tables



    OUTPUT:
    ---
    - processed_data/events/event3: only dishoest person blacklist
    - processed_data/meta_events/crosstab_event3
        - cross table for company and categoreis
    - processed_data/crosstab_year_event3
        - cross table for company, year and categoreis
    - processed_data/crosstab_day_event3
        - day-level cross table for ID, company, date and #blacklist

    PROCESS:
    ---
    - fill start years from content for safety production & tax blacklist
    - cross table for blacklist types
    - Then focus on dishonest person only
        - add categories:
            - level(from base and court)
            - theme
            - inclusion reason
        - crosstabs(year)
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# # all blacklists cross table

# In[2]:


def load_data():
    df_event_dishonest_person = pd.read_excel(os.path.join(sub_event_path, '31_event_dishonest_person.xlsx'))
    df_event_safety_production = pd.read_excel(os.path.join(sub_event_path, '32_event_safety_production.xlsx'))
    df_event_tax_blacklist = pd.read_excel(os.path.join(sub_event_path, '33_event_tax_blacklist.xlsx'))
    df_event_gov_procure_illegal = pd.read_excel(os.path.join(sub_event_path, '34_event_gov_procure_illegal.xlsx'))
    df_event_overload_transport_illegal = pd.read_excel(os.path.join(sub_event_path, '35_event_overload_transport_illegal.xlsx'))
    
    return df_event_dishonest_person, df_event_safety_production, \
        df_event_tax_blacklist, df_event_gov_procure_illegal, df_event_overload_transport_illegal


# In[3]:


def get_blacklist_crosstabs(
    df_event_tax_blacklist,
    df_event_safety_production,
    df_event_gov_procure_illegal,
    df_event_overload_transport_illegal,
    df_event_dishonest_person
):
    """
    Get blacklist-related crosstab DataFrames.
    
    Parameters
    ----------
    df_event_tax_blacklist : DataFrame
    df_event_safety_production : DataFrame
    df_event_gov_procure_illegal : DataFrame
    df_event_overload_transport_illegal : DataFrame
    df_event_dishonest_person : DataFrame
    
    Returns
    -------
    df_event_blacklists : DataFrame
        Contains 4 columns: company_name, start_year, end_year, blacklist_type
    df_crosstab_blacklists : DataFrame
        Contains columns: blacklist, Blacklist_type:XXX, and MultiIndex: company_name, ID
    df_crosstab_year_blacklists : DataFrame
        Contains columns: blacklist, Blacklist_type:XXX, and MultiIndex: company_name, ID, year
    df_crosstab_day_blacklists : DataFrame
        Contains columns: ID, company_name, date #blacklists
    """
    df_event_blacklists_l = []

    # append each blacklist
    df_event_blacklists_l.append(get_uniform_table(df_event_tax_blacklist,
                     'start_year',
                     'start_date',
                     -1,
                     'Tax blacklist',
                     'blacklist_type'
                    ))

    df_event_blacklists_l.append(get_uniform_table(df_event_safety_production,
                     'start_year',
                     'start_date',
                     -1,
                     'Safety production blacklist',
                     'blacklist_type'
                    ))

    df_event_blacklists_l.append(get_uniform_table(df_event_gov_procure_illegal,
                     'start_year',
                     'start_date',
                     -1,
                     'Government procurement blacklist',
                     'blacklist_type'
                    ))

    df_event_blacklists_l.append(get_uniform_table(df_event_overload_transport_illegal,
                     'start_year',
                     'start_date',
                     -1,
                     'Illegal overload blacklist',
                     'blacklist_type'
                    ))

    df_event_blacklists_l.append(get_uniform_table(df_event_dishonest_person,
                     'issue_year',
                     'issue_date',
                     -1,
                     'Dishonest_person blacklist',
                     'blacklist_type'
                    ))

    # concat
    df_event_blacklists = pd.concat(df_event_blacklists_l)
    
    # day-level crosstab
    no_col_l = ['blacklist_tax', 
                'blacklist_safety_production', 
                'blacklist_gov_procure_illegal', 
                'blacklist_overload_transport_illegal', 
                'blacklist_dishonest_person'
               ]
    
    # get df_period for each event
    df_period_l = []
    for df_event, no_col in zip(df_event_blacklists_l, no_col_l):
        df_period_l.append(get_period_df(df_event, no_col))
    
    # concat
    df_period = concat_periods(df_period_l, 'blacklist')
    display(df_period)

    # crosstab
    df_crosstab_blacklists = get_crosstab(df_event_blacklists, 'Blacklists:', 'blacklist_type')

    # add total number
    df_crosstab_blacklists = add_margins(df_crosstab_blacklists, 'blacklist')
    
    display(df_crosstab_blacklists.head())

    # crosstab-year
    df_crosstab_year_blacklists = get_crosstab_by_year(
        df_event_blacklists, 'Blacklist_Type:', 'blacklist_type', 'start_year', 'end_year')

    df_crosstab_year_blacklists = add_margins(df_crosstab_year_blacklists, 'blacklist')

    display(df_crosstab_year_blacklists.head())
    
    return df_event_blacklists, df_crosstab_blacklists, df_crosstab_year_blacklists, df_period


# # dishonest person blacklist

# ## Category1: Reason for inclusion(Multi-choice category)
# - 有履行能力而拒不履行生效法律文书确定义务的
#     - search keyword:有履行能力
#     - Have the ability to perform and refused to perform the obligations determined by the effective legal instruments
#     - validation years: Most severe type, no specific years
# - 以伪造证据、暴力、威胁等方法妨碍、抗拒执行的
#     - search keyword: 抗拒执行
#     - Obstruction and resistance to execution by falsification of evidence, violence, threats and other methods
#     - validation years: Second severe type, generally 2 years, severe can add 1-3 years
# - 以虚假诉讼、虚假仲裁或者以隐匿、转移财产等方法规避执行的；
#     - search keyword:规避执行
#     - Evade execution by false lawsuit, false arbitration or by concealing or transferring property
#     - validation years: 2 years
# - 违反财产报告制度的；
#     - search keyword: 财产报告
#     - Violation of the property reporting system
#     - validation years: 2 years
# - 违反限制消费令的；
#     - search keyword: 限制消费
#     - Violation of consumption restriction order
#     - validation years: 2 years
# - 无正当理由拒不履行执行和解协议的 
#     - search keyword:无正当理由
#     - Refusal to perform the settlement agreement without valid reasons
#     - validation years: 2 years

# In[4]:


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
        '有履行能力':'A-Have the ability but refuse to perform the obligations',
        '抗拒执行':'B-Resisting enforcement through falsification of evidence/violence/threats/etc',
        '规避执行':'C-Evade execution through false lawsuits/false arbitration/concealment or transfer of property/etc',
        '财产报告':'D-Violation of the property reporting system',
        '限制高消费':'E-Violation of consumption restriction order',
        '无正当理由': 'F-Refusal to perform the settlement agreement without valid reasons',
        '— —' : 'NA'
    }
    
    return add_multi_choice_category_from_keywords(
        df_event, 
        'inclusion_reason',
        '失信被执行人行为具体情形 Person of Execution Untrustworthy Behavior Details',
        inclusion_reason_dict)


# ## category2: Theme from execution base unit
# 
# - Intellectual Property
# - Transportation
# - Notary Public Office
# - HR
# - Courts

# In[5]:


def add_themes(df_event):
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
    theme_dict = {
        '知识产权':'Intellectual Property',
        '铁路运输法院':'Transportation',
        '海事法院':'Transportation',
        '公证处':'Notary Public Office',
        '劳动':'HR',
        '人事': 'HR',
        '人力资源' : 'HR',
        '人社局' : 'HR',
        '劳人仲' : 'HR',
        '仲裁' : 'HR',
        '人力资源' : 'HR'        
    }
    
    df_event = add_single_choice_category_from_keywords(
        df_event, 
        'theme',
        '做出执行依据单位 Execution Base Unit',
        theme_dict)
    
    # set default 
    df_event.theme = df_event.theme.fillna('Courts')
    
    return df_event


# In[6]:


def handle_region(df_event):
    """
    Extract region, authority info from authority and fill 1 missing province with the extracted province.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        df_event with columns: province, level, authority.
    """
    df_auth = extract_authorites(df_event['做出执行依据单位 Execution Base Unit'])[['province', 'level', 'authority']]
    df_region = extract_regions(df_event['执行法院 Execution Court'])['level']
    
    # fill province
    na_id = df_event[df_event['省份  Province'] == '— —'].index
    
    df_event.loc[na_id, '省份  Province'] = df_auth.loc[na_id, 'province']
    
    # fill NA level in region by level in df_auth
    na_id = df_region[df_region == 'uncertein'].index
    
    df_region[na_id] = df_auth.loc[na_id, 'level']
    
    # fill NA court, but since we do not use court, so not run this part
    #na_id = df_event[df_event['执行法院 Execution Court'] == '??ī???????'].index
    #df_event.loc[na_id, '执行法院 Execution Court'] = df_event.loc[na_id, '做出执行依据单位 Execution Base Unit']
    
    # fill second time by area(manually checked the left 15 uncertain are all belong to area)
    df_region[df_region[df_region == 'uncertein'].index] = 'area'
    
    return pd.concat([df_event, df_region, df_auth['authority']], axis=1).rename(
        columns = {
            '省份  Province': 'province'
        })


# In[13]:


def process_event():
    """
    Get crosstabs(& year) for blacklists by processing 5 blacklist-related sub events.
    
    Returns
    -------
    df_crosstab_blacklists : DataFrame
        A DataFrame contains cross tale for 1 blacklists and 3 categories of dishonest person with MultiIndex(company_name, ID)
    df_crosstab_year_blacklists : DataFrame
        A DataFrame with MultiIndex(company_name, ID, year)
    df_period : DataFrame
        Data from each time point.
    """
    print('load data...')

    df_event_dishonest_person, df_event_safety_production, \
    df_event_tax_blacklist, df_event_gov_procure_illegal, df_event_overload_transport_illegal = load_data()

    print('get crosstabs(including year, day-level) for 5 types of blacklists...')
    df_event_blacklists, df_crosstab_blacklists, df_crosstab_year_blacklists, df_crosstab_day_blacklists = get_blacklist_crosstabs(
        df_event_tax_blacklist,
        df_event_safety_production,
        df_event_gov_procure_illegal,
        df_event_overload_transport_illegal,
        df_event_dishonest_person
    )

    # add 3 category columns
    print('add 3 category columns...')
    df_event_dishonest_person = df_event_dishonest_person.pipe(
        add_inclusion_reasons).pipe(
        add_themes).pipe(handle_region)
    
    # handle unknown provinces    
    df_event_dishonest_person = standard_province_names(df_event_dishonest_person)

    # save event
    print('save processed df_event...')
    df_event_dishonest_person[['company_name', 
                           'province', 
                           'issue_year', 'issue_date', 
                           'inclusion_reason', 
                           'theme', 
                           'level', 
                           'authority', 
                           '生效法律文书确定的义务 Obligations Determined by Effective Legal Documents', 
                           '被执行人的履行情况 Implementation Performance of Person of Execution',
                           '已履行部分 Part of Accomplishment',
                           '未履行部分 Part of Non-accomplishment'
                           ]].to_excel(
        os.path.join(processed_event_path, 'event3.xlsx'), index=False, freeze_panes=(1,2))

    # draw distributions
    draw_distribution(
        [df_event_dishonest_person.province, df_event_dishonest_person.level, df_event_dishonest_person.inclusion_reason, 
         df_event_dishonest_person.theme, df_event_dishonest_person.issue_year],
        ['pie', 'pie', 'pie', 'pie', 'pie']
    )

    # crosstab
    print('generate cross tables of campany and categories for Dishonest person blacklist...')

    cat_prefix = ['BL_DP_level:', 'BL_DP_theme:', 'BL_DP_inclustion_reason:']
    cat_column_names = ['level', 'theme', 'inclusion_reason']
    number_column_name = 'dishonest_person_number'
    multi_flags = [False, False, True] # single-choince, single, multi-choice

    df_crosstab_dishonest_person = get_crosstabs_for_multi_categories(
        df_event_dishonest_person, 
        cat_prefix, 
        cat_column_names, 
        number_column_name, 
        multi_flags)

    display(df_crosstab_dishonest_person.head())

    # combine with crosstab of blacklists
    df_crosstab_blacklists = pd.concat(
        [df_crosstab_blacklists, df_crosstab_dishonest_person], axis=1)

    # cross tab by year
    print('generate cross tables of [campany, year] and categories for Dishonest person blacklist...')

    df_crosstab_year_dishonest_person = get_crosstabs_for_multi_categories_by_year(
        df_event_dishonest_person, cat_prefix, 
        cat_column_names, 
        number_column_name, 
        multi_flags, 
        'issue_year')

    display(df_crosstab_year_dishonest_person.head())

    # combine with crosstab year of blacklists
    df_crosstab_year_blacklists = pd.concat(
        [df_crosstab_year_blacklists, df_crosstab_year_dishonest_person], axis=1)
    
    # save
    save_meta_events(df_crosstab_blacklists, df_crosstab_year_blacklists, df_crosstab_day_blacklists, 'event3')
    
    return df_crosstab_blacklists, df_crosstab_year_blacklists, df_crosstab_day_blacklists


# process_event()

# In[ ]:




