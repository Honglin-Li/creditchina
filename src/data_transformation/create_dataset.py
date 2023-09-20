#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
create_dataset

The module provides functions to create the final dataset and sample data.
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

#from utils import processed_data_path, original_path
from src.data_transformation.utils import *
import os
import src.data_transformation.event1_permit_processing as event1_permit
import src.data_transformation.event1_penalty_processing as event1_penalty
import src.data_transformation.event2_processing as event2
import src.data_transformation.event3_processing as event3
import src.data_transformation.event4_processing as event4
import src.data_transformation.event5_processing as event5
from src.data_transformation.split_events import split_events
from src.data_transformation.pipeline import *
from src.data_transformation.event_stat import *


# In[2]:


def create_credit_dataset():
    """
    Create the credit dataset from the extracted data.
    
    This evolvs split, pre-processing, and categorize events. Then create metadata(year- and day-level) based on the cleaned events.
    The data set is verified by comparing with original metadata.
    """
    # check if the data is downloaded and extracted:
    if not os.path.exists(os.path.join(original_path, 'mothers_original.xlsx')):
        print('You should download the credit reports and extract the data first before you create data set!!!')
        return
        
    # set parameters
    # event column name
    file_name_list = ['event1_permit', 'event1_penalty', 'event2', 'event3', 'event4', 'event5']

    # func to generate processed events & crosstabs
    func_list = [event1_permit.process_event,
                 event1_penalty.process_event,
                 event2.process_event, 
                 event3.process_event, 
                 event4.process_event,
                 event5.process_event]

    # 1. split and pre-process events
    meta_path = os.path.join(processed_data_path, 'metadata.xlsx')
    
    if not os.path.exists(meta_path):
        split_events()
    
    # remove the incorrect event records
    #remove_incorrect_event_records(pd.read_excel(meta_path))

    # 2. get crosstabs
    print('generating crosstabs for each events...')
    crosstabs, crosstabs_year, crosstabs_day = get_crosstab_dataframes(file_name_list, func_list)

    # 3. create final dataset
    df_meta = get_meta_table(crosstabs, new=False)
    df_meta_year = get_meta_year_table(df_meta, crosstabs_year, new=True)
    df_meta_day = get_meta_day_table(df_meta, crosstabs_day, new=True)
    
    # 4. internal check: START YEAR V.S. FOUNDATION YEAR
    print('internal consistency check...')
    
    df_result = internal_consistency_check_year(df_meta, df_meta_year)
    
    if df_result.shape[0] > 0:
        print('year internal consistency check failed!!!')
        #display(df_result.head(50))
        return df_result
    
    # 5. credit rating variations
    check_rating_variation_types(df_meta_year)
    create_metadata_year_for_variation_firms(df_meta_year)
    
    print('credit category variation tables over time (all, mother, daughter)')
    print(get_rating_variation_table(df_meta_year))
    
    df_meta_year = add_M_D_column(df_meta_year, df_meta)
    print(get_rating_variation_table(df_meta_year[df_meta_year.M_D != 'D']))
    print(get_rating_variation_table(df_meta_year[df_meta_year.M_D == 'D']))
    
    
    # 6. internal check: compare actual and predicted CREDIT CATEGORY
    internal_consistency_check_credit_category(df_meta, df_meta_year)
    internal_consistency_check_credit_category(df_meta, df_meta_day, 'day')
    
    # 7. performance data
    df_performance_panel = get_performance_panel_data(df_meta)
    
    df_combined_panel = get_performance_credit_panel_data(df_performance_panel, df_meta_year) # for regression
    
    # 8. mother daughter relation & stat
    df_relation_panel = get_mother_daughter_relation_panel_data(df_meta, df_meta_year)
    
    df_relation_stat_panel = get_mother_daughter_stat_panel_data(df_relation_panel)
    
    # 9. final credit-performance-relation table, for regression considering daughters
    df_final_panel = get_performance_credit_relationship_panel_data(df_combined_panel, df_relation_stat_panel)
    
    


# # analysis

# In[3]:


def credit_analysis():
    """
    Generate credit tables and reports for analysis.
    
    The generated analysis is in the data/stat directory.
    """    
    # load data
    # meta
    df_meta = get_credit_meta_data().reset_index()

    no_company = df_meta.shape[0]

    # load event
    file_name_list = ['event1_permit', 'event1_penalty', 'event2', 'event3', 'event4', 'event5']

    event1_permit, event1_penalty, event2, event3, event4, event5 = add_M_D_column_for_list(
        get_event_dataframes(file_name_list),
        df_meta
    ) # add M_D

    crosstabs, crosstabs_year, crosstabs_day = get_crosstab_dataframes(file_name_list, [0,0,0,0,0,0])

    # load crosstabs
    ct1_permit, ct1_penalty, ct2, ct3, ct4, ct5 = add_M_D_column_for_list(
        crosstabs,
        df_meta
    )

    ct1_year_permit, ct1_year_penalty, ct2_year, ct3_year, ct4_year, ct5_year = add_M_D_column_for_list(
        crosstabs_year,
        df_meta
    )

    #ct1_day_permit, ct1_day_penalty, ct2_day, ct3_day, ct4_day, ct5_day = crosstabs_day
    # generate stat for metadata(all firms/ mothers/ daughters)
    df_province, df_province_year, df_relation, df_year_company = generate_meta_stat(df_meta, 'stat_meta.xlsx')

    df_province_m, df_province_year_m, df_relation_m, df_year_company_m = generate_meta_stat(
        df_meta[df_meta.M_D!='D'], 'stat_meta_mothers.xlsx'
    )

    df_province_d, df_province_year_d, df_relation_d, df_year_company_d = generate_meta_stat(
        df_meta[df_meta.M_D=='D'], 'stat_meta_daughters.xlsx'
    )

    # generate for each stat
    generate_event_stat(df_year_company_m, df_year_company_d, event1_permit, ct1_year_permit, 'permit', ['permit_type', 'theme', 'level'], 'start_year', 'end_year')

    generate_event_stat(df_year_company_m, df_year_company_d, event1_penalty, ct1_year_penalty, 'penalty', ['penalty_type', 'theme', 'level'])

    generate_event_stat(df_year_company_m, df_year_company_d,  event2, ct2_year, 'redlist', ['redlist_type'])

    generate_event_stat(df_year_company_m, df_year_company_d,  event3, ct3_year, 'blacklist', ['inclusion_reason', 'theme', 'level'], 'issue_year')

    generate_event_stat(df_year_company_m, df_year_company_d,  event4, ct4_year, 'watchlist', ['inclusion_reason', 'level'])

    generate_event_stat(df_year_company_m, df_year_company_d,  event5, ct5_year, 'commit', ['commit_type', 'theme', 'level'])



# # Sampling for thesis data submission

# In[ ]:


def get_sampled_daughter_list(m_list, sample_count=10):
    """
    Get sampled daughter name list by mother name list.
    
    Parameters
    ----------
    m_list : list of str
        A list of parent companies to sample.
    sample_count : int, default 10
        The sample count for each mother company
    
    Returns
    -------
    list of str
        The list of sampled subsidiry names.
    """
    df_r = get_mother_daughter_relation_panel_data()
    df_r = df_r[df_r.mother_name.isin(m_list)]
    df_r = df_r[['mother_name', 'daughter_name']].drop_duplicates()
    
    def sample(df):
        if df.shape[0] > sample_count:
            return df.sample(sample_count)
        else:
            return df

    return df_r.groupby('mother_name', as_index=False).apply(sample).daughter_name.tolist()


# create_credit_dateset()

# In[66]:


def create_sample_for_original(m_list, d_list):
    """
    Create and save the sample original mothers and corresponding daughters.
    
    Parameters
    ----------
    m_list : list of str
        A list of parent companies to sample.
    """
    def get_sample_event(xls, event_name, sample_companies):
        df_event = pd.read_excel(xls, event_name)
        return df_event[df_event.company_name.isin(sample_companies)]
    
    def create_sample(ori_path, tar_path, company_list):
        # read and sample
        with pd.ExcelFile(ori_path) as original:
            # sampled mothers
            df_meta = pd.read_excel(original, 'metadata')
            df_meta = df_meta[df_meta[company_name].isin(company_list)]

            df_event1 = get_sample_event(original, 'event1', company_list)
            df_event2 = get_sample_event(original, 'event2', company_list)
            df_event3 = get_sample_event(original, 'event3', company_list)
            df_event4 = get_sample_event(original, 'event4', company_list)
            df_event5 = get_sample_event(original, 'event5', company_list)
        
        # save
        with pd.ExcelWriter(tar_path) as writer:
            df_meta.to_excel(writer, sheet_name='metadata', index=False, freeze_panes=(1, 2))
            df_event1.to_excel(writer, sheet_name='event1', index=False, freeze_panes=(1, 1))
            df_event2.to_excel(writer, sheet_name='event2', index=False, freeze_panes=(1, 1))
            df_event3.to_excel(writer, sheet_name='event3', index=False, freeze_panes=(1, 1))
            df_event4.to_excel(writer, sheet_name='event4', index=False, freeze_panes=(1, 1))
            df_event5.to_excel(writer, sheet_name='event5', index=False, freeze_panes=(1, 1))
    
    # original path
    m_ori_path = os.path.join(data_dir_path, 'original', 'mothers_original.xlsx')
    d_ori_path = os.path.join(data_dir_path, 'original', 'daughters_original.xlsx')
    
    # sample mothers
    create_sample(m_ori_path, os.path.join(original_path, 'mothers_original_sample.xlsx'), m_list)
    
    # sample daughters
    create_sample(d_ori_path, os.path.join(original_path, 'daughters_original_sample.xlsx'), d_list)

    


# In[67]:


#create_sample_for_original(m_50, d_list)


# In[ ]:




