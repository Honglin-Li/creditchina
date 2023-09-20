#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
add_theme

A module to add theme classifications for observations without authorities in permit, penalty, and commit events.
"""


# In[1]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.theme_classification import prepare_permit_ds, prepare_penalty_ds, prepare_commitment_ds, save_event_with_theme, predict
from src.data_transformation.utils import draw_bar, sub_event_path
import pandas as pd
import os


# In[9]:


def add_penalty_themes(df_event_penalty=None):
    """
    Add theme column to the penalty event by classification with the tuned hyperparameters, then save to local.
    
    Parameters
    ----------
    df_event_penalty ：DataFrame, optional
    
    Returns
    -------
    DataFrame
        The DataFrame event with theme columns.
    """
    if df_event_penalty is None:
        df_event_penalty = pd.read_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'))
    
    ds_train, ds_unseen, df_train, df_test = prepare_penalty_ds(df_event_penalty)

    df_unseen = predict(ds_unseen, 'penalty_model.pickle')

    print('Distribution of unseen records')
    draw_bar(df_unseen.theme)

    # save to local
    return save_event_with_theme(df_event_penalty, ds_train, df_unseen, '12_event_penalty.xlsx')
    


# In[22]:


def add_commitment_themes(df_event_commitment=None, t=0.65):
    """
    Add theme column to the commitment event by classification with the tuned hyperparameters, then save to local.
    
    Parameters
    ----------
    df_event_commitment ：DataFrame, optional
    t : float, default 0.65
        The probablity threshold of predicting.
    
    Returns
    -------
    DataFrame
        The DataFrame event with theme columns.
    """
    if df_event_commitment is None:
        df_event_commitment = pd.read_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'))
    
    ds_train, ds_unseen, df_train, df_test = prepare_commitment_ds(df_event_commitment)
    df_unseen = predict(ds_unseen, 'commit_model.pickle', t)

    print('Distribution of unseen records')
    draw_bar(df_unseen.theme)

    # save to local
    return save_event_with_theme(df_event_commitment, ds_train, df_unseen, '51_event_commitment_implementation.xlsx')
    


# In[18]:


def add_permit_themes(df_event_permit=None, t=0.75):
    """
    Add theme column to the permit event by classification with the tuned hyperparameters, then save to local.
    
    Parameters
    ----------
    df_event_permit ：DataFrame, optional
    t : float, default 0.65
        The probablity threshold of predicting.
    
    Returns
    -------
    DataFrame
        The DataFrame event with theme columns.
    """
    if df_event_permit is None:
        df_event_permit = pd.read_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'))
    
    ds_train, ds_unseen, df_train, df_test = prepare_permit_ds(df_event_permit)
    df_unseen = predict(ds_unseen, 'permit_model.pickle', t) 

    print('Distribution of unseen records')
    draw_bar(df_unseen.theme)

    # save to local
    return save_event_with_theme(df_event_permit, ds_train, df_unseen, '11_event_permit.xlsx')


# In[ ]:




