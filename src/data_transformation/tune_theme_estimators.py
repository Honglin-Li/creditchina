#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
from theme_classification import *


# ## Tune Event: penalty
# 
# Best parameter settings:
# - min_df: 2
# - max_df: 1000
# - alpha: 0.1
# - t: none
# - s: 1000

# In[ ]:


#-----------------------1. check WORD CLOUD -------------------
print('prepare data source[content, theme]...')

df_event_penalty = pd.read_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'))

ds_train, ds_unseen, df_train, df_test = prepare_penalty_ds(df_event_penalty)

#show_wordcloud_for_event(ds_train, 'Penalty ')


# In[ ]:


#-----------------------2. Tuning: GET the best Parameter values-------------------
# parameter options
s_count_l = [100, 700, 1000]
min_df_l = [3, 5]
max_df_l = [100, 1000]
alpha_l = [0.0001, 0.01, 0.5]
t_l = []

tune_hyperparameters(df_train, t_l, min_df_l, max_df_l, alpha_l, s_count_l)


# In[ ]:


#-----------------------3. get generalization performance-------------------
generalization_performance(df_train, df_test, 0, 2, 1000, 0.1, 1000)


# In[ ]:


# save model
save_model(ds_train, 'penalty_model.pickle', 0, 2, 1000, 0.1, 1000)
#df_unseen = predict(ds_train, ds_unseen, 'penalty_model.pickle')
#save_event_with_theme(df_event_penalty, ds_train, df_unseen, '12_event_penalty.xlsx')


# ## Tune Event: permit
# Best parameter settings:
# - min_df: 4
# - max_df: 0.9
# - alpha: 0.05
# - t: 0.6
# 
# - acc 94%/98.77%

# In[2]:


#-----------------------1. check WORD CLOUD -------------------
print('prepare data source[content, theme]...')

df_event_permit = pd.read_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'))

ds_train, ds_unseen, df_train, df_test = prepare_permit_ds(df_event_permit)

#show_wordcloud_for_event(ds_train, 'Permit ')


# # prepare_permit_ds() needs to much time to run
# save_dir = os.path.join(data_path, 'models', 'train')
# 
# #ds_train.to_excel(os.path.join(save_dir, 'permit_ds_train.xlsx'), index=False)
# #ds_unseen.to_excel(os.path.join(save_dir, 'permit_ds_unseen.xlsx'), index=False)
# #df_train.to_excel(os.path.join(save_dir, 'permit_df_train.xlsx'), index=False)
# #df_test.to_excel(os.path.join(save_dir, 'permit_df_test.xlsx'), index=False)
# 
# ds_train = pd.read_excel(os.path.join(save_dir, 'permit_ds_train.xlsx'))
# ds_unseen = pd.read_excel(os.path.join(save_dir, 'permit_ds_unseen.xlsx'))
# df_train = pd.read_excel(os.path.join(save_dir, 'permit_df_train.xlsx'))
# df_test = pd.read_excel(os.path.join(save_dir, 'permit_df_test.xlsx'))

# In[10]:


#-----------------------2. Tuning: GET the best Parameter values-------------------
# parameter options
s_count_l = []
min_df_l = []
max_df_l = [1500, 2500]
alpha_l = []
t_l = []

tune_hyperparameters(df_train, t_l, min_df_l, max_df_l, alpha_l, s_count_l)


# In[32]:


#-----------------------3. get generalization performance-------------------
generalization_performance(df_train, df_test, 0.75, 3, 1.0, 0.1, 1000)


# In[37]:


# save model
#save_model(ds_train, 'permit_model_without_threshold.pickle', 0, 3, 1.0, 0.1, 1000)
df_unseen = predict(ds_train, ds_unseen, 'permit_model.pickle', 0.75)
save_event_with_theme(df_event_permit, ds_train, df_unseen, '11_event_permit.xlsx')


# ## Tune Event: Commitment
# Best parameter settings:
# - min_df: 2
# - max_df: 500
# - alpha: 0.5
# - t: 0.65 or none
# 
# 

# In[ ]:


#-----------------------1. check WORD CLOUD -------------------
print('prepare data source[content, theme]...')

df_event_commitment = pd.read_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'))

ds_train, ds_unseen, df_train, df_test = prepare_commitment_ds(df_event_commitment)

#show_wordcloud_for_event(ds_train, 'Commit ')


# In[ ]:


#-----------------------2. Tuning: GET the best Parameter values-------------------
# parameter options
s_count_l = [500, 1000]
min_df_l = [2, 7]
max_df_l = [100, 500, 1000]
alpha_l = [0.0001, 0.01, 0.5]
t_l = [0, 0.5]

tune_hyperparameters(df_train, t_l, min_df_l, max_df_l, alpha_l, s_count_l)


# In[ ]:


#-----------------------3. get generalization performance-------------------
generalization_performance(df_train, df_test, 0, 2, 500, 0.5, 650)


# In[ ]:


# with threshold
print(df_test.shape)
generalization_performance(df_train, df_test, 0.65, 2, 500, 0.5, 650)


# In[ ]:


# save the best model
save_model(ds_train, 'commit_model.pickle', 0.65, 2, 500, 0.5, 650)
#df_unseen = predict(ds_train, ds_unseen, 'commit_model.pickle')
#df = save_event_with_theme(df_event_commitment, ds_train, df_unseen, '51_event_commitment_implementation.xlsx')


# In[ ]:


# check unknown
df.theme.value_counts()


# In[ ]:




