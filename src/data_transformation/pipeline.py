#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
pipeline

Provide all the functions to create the final dataset & Internal consistency check.
"""


# In[2]:


# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))


# In[3]:


from src.data_transformation.utils import *
import src.data_transformation.event1_permit_processing as event1_permit
import src.data_transformation.event1_penalty_processing as event1_penalty
import src.data_transformation.event2_processing as event2
import src.data_transformation.event3_processing as event3
import src.data_transformation.event4_processing as event4
import src.data_transformation.event5_processing as event5
from src.data_transformation.split_events import split_events


# # Credit dataset set up pipeline

# In[4]:


def get_crosstab_dataframes(file_name_list, func_list):
    """
    Get crosstab(year) and day-level DataFrames.
    
    This func wont run by calling get_meta_year_table. If you want to re-create those crosstabs for events, \
    you need to delete the files in local first.
    
    Parameters
    ----------
    file_name_list : list of str
        The item is the file name without extension, like event1.
    func_list : list of functions
        The func is to create crosstabs if the crosstabs not in local.
    
    Returns
    ----
    crosstabs : list of DataFrames
    crosstabs_year : list of DataFrames
    crosstabs_day : list of DataFrames
    """   
    crosstabs = []
    crosstabs_year = []
    crosstabs_day = []
    
    for file, func in zip(file_name_list, func_list):
        # get file name of crosstabs
        crosstab_path, crosstab_year_path, crosstab_day_path = get_meta_event_paths(file)
        
        # not exists -> create
        if not os.path.exists(crosstab_day_path):
            # call responding function to generate
            print(f'create crosstabs for {file}...')
            df_crosstab, df_crosstab_year, df_crosstab_day = func()
            
        else:
            df_crosstab = pd.read_excel(crosstab_path, index_col=[0, 1])
            df_crosstab_year = pd.read_excel(crosstab_year_path, index_col=[0, 1, 2])
            df_crosstab_day = pd.read_excel(crosstab_day_path, index_col=[0, 1, 2])
        
        crosstabs.append(df_crosstab)
        crosstabs_year.append(df_crosstab_year)
        crosstabs_day.append(df_crosstab_day)
    
    return crosstabs, crosstabs_year, crosstabs_day


# In[5]:


def get_event_dataframes(file_name_list):
    """
    Get list of processed event DataFrame.
    
    Parameters
    ----------
    file_name_list : list of str
        The item is the file name without extension, like event1.
    
    Returns
    ----
    list of DataFrames
        The item is processed sub-event DataFrame.
    """   
    results = []
    
    for file in file_name_list:
        path = os.path.join(processed_event_path, file + '.xlsx')
        
        # check exists
        if not os.path.exists(path):
            print(f'{file} does not exists, you can call function get_crosstab_dataframes() to create processed events and crosstabs.')
            break
            
        results.append(
            pd.read_excel(path)
        )
    
    return results


# In[6]:


def add_total_event_numbers(df):
    """
    Add total event numbers for meta.
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    DataFrame
    """
    df['event_number'] = df[[
        'permit',
        'penalty',
        'redlist', 
        'blacklist', 
        'watchlist',
        'commit'
    ]].sum(axis=1) 
    
    df = move_last_column_to_first(df)
    
    return df


# In[7]:


def label_companies(df_meta, metadata, data_type):
    """
    Assign red/black/green to each company each record of DataFrame df_meta.
    
    Parameters
    ----------
    df_meta : DataFrame
        The DataFrame can be crosstab_year or crosstab_day.
    metadata : DataFrame
        The main metadata which contains company_name, ID, and foundation_year.
    date_type : str
        Can be 'year' or 'day'
    
    Returns
    -------
    DataFrame
        With additional red, black, green columns.
    """
    # 1. Label
    def label_black_red(df_meta):
        # set black. RULE: in blacklists
        df_meta['black'] = (df_meta.blacklist > 0).astype(int)

        # set red. RULE: in redlists & not in blacklists
        df_meta['red'] = ((df_meta.redlist > 0) & (df_meta.black == 0)).astype(int)
    
    label_black_red(df_meta)
    
    if data_type == 'year':
        df_meta = handle_green_credit_rating(df_meta, metadata)
        
    else:
        # for metadata_day, only add green label
        df_meta['green'] = 0
        
    # assign green to NA
    green_idx = df_meta[(df_meta['red'] == 0) & (df_meta['black'] == 0)].index
    df_meta.loc[green_idx, 'green'] = 1
    df_meta['green'] = df_meta['green'].fillna(0)
    
    label_black_red(df_meta) # in case sth wrong, label one more time
    
    # 2. Check the correctness of credit category
    def show_incorrect_credit_category(color, incorrect_idx):
        print(f'check credit category: {color}...')
        print(incorrect_idx.sum())
        
        if incorrect_idx.sum() > 0:
            print(df_meta[incorrect_idx].head(50))
        
    incorrect_idx = (df_meta.blacklist > 0) & (df_meta.black != 1)
    show_incorrect_credit_category('black', incorrect_idx)
    
    incorrect_idx = (df_meta.blacklist == 0) & (df_meta.redlist > 0) & (df_meta.red != 1)
    show_incorrect_credit_category('red', incorrect_idx)
    
    incorrect_idx = (df_meta.blacklist == 0) & (df_meta.redlist == 0) & (df_meta.green != 1)
    show_incorrect_credit_category('green', incorrect_idx)
    
    print('check if green+black+red=1...')
    print(df_meta[['black', 'red', 'green']].sum(axis=1).value_counts())
    
    return df_meta


# In[8]:


def handle_green_credit_rating(df_meta, metadata):
    """
    Assign green rating for all companies all years (new records are added).
    
    Parameters
    ----------
    df_meta : DataFrame
        The DataFrame can be crosstab_year or crosstab_day.
    metadata : DataFrame
        The main metadata which contains company_name, ID, and foundation_year.
    
    Returns
    -------
    DataFrame
        With additional green column and rows (companies without any event records).
    """
    df_source = metadata.reset_index()[['ID', 'company_name', 'foundation_year']]
    
    # Type1: companies without any event records, so that not in df_meta
    companies = list(set(df_source.company_name.unique()) - set(df_meta.index.unique('company_name')))
    
    print(f'TYPE1: {len(companies)} companies do not have any event records.')
    
    df_new_companies = df_source[df_source['company_name'].isin(companies)]
    
    # handle the year
    df_new_companies = prepare_year_column(df_new_companies, 'ID', 'foundation_year') # columns: company_name, ID, year
    
    # Type2: companies in df_meta but the start year is not equal to the foundation year
    df_exist_companies = df_meta.reset_index().groupby('company_name')['year'].min()
    df_exist_companies = df_exist_companies.to_frame().join(df_source.set_index('company_name'), how='left') # add foundation year
    df_exist_companies['end_year'] = df_exist_companies.year - 1
    
    # remove the companies with start year equal to 2014
    df_exist_companies = df_exist_companies[df_exist_companies.year > s_year].reset_index()
    
    # handle the year
    df_exist_companies = prepare_year_column(df_exist_companies, 'ID', 'foundation_year', 'end_year')
    
    # Type3: companies in df_meta but the end year is not equal to 2022 (not have to be green)
    df_exist_companies_end = df_meta.reset_index().groupby('company_name')['year'].max().to_frame()
    
    df_exist_companies_end = df_exist_companies_end[df_exist_companies_end.year < e_year].reset_index()
    
    df_exist_companies_end['start_year'] = df_exist_companies_end.year + 1
    
    # combine the 2 types of companies
    df_new_records = pd.concat([df_new_companies, df_exist_companies])
    
    # add credit rating columns
    df_new_records['red'] = 0
    df_new_records['black'] = 0
    df_new_records['green'] = 1
    
    if df_exist_companies_end.shape[0] > 0:
        # handle the year
        df_exist_companies_end = prepare_year_column(assign_id_to_company(df_exist_companies_end), 'ID', 'start_year')
        
        # combine the type 3
        df_new_records = pd.concat([df_new_records, df_exist_companies_end]) # credit category columns will be NA
    
    df_new_records = df_new_records.set_index(['ID', 'company_name', 'year'])
    
    df_meta = pd.concat([df_meta.reset_index(), df_new_records.reset_index()]) # concat new rows
    
    # remove the incorrect observations before foundation date.
    df_meta = df_meta.reset_index(drop=True).merge(df_source[['ID', 'foundation_year']], how='left', on='ID') # add foundation year
    
    remove_idx = df_meta[df_meta.year < df_meta.foundation_year].index
    print(f'{len(remove_idx)} observations are earlier than foundation year')
    
    df_meta = df_meta.drop(
        remove_idx
    ).drop(
        ['foundation_year'], 
        axis=1
    ).sort_values(
        ['ID', 'year']
    ).set_index(
        ['ID', 'company_name', 'year']
    )
    
    # handle Type3's credit category
    df_meta[['red', 'black']] = df_meta[['red', 'black']].fillna(method='ffill') # green can not fill NA
    
    return df_meta
    


# In[9]:


def get_meta_table(crosstabs=None, new=False):
    """
    Load from local or Create Metadata by merging descriptive columns in df_meta and crosstabs from events.
    
    Parameters
    ----------
    crosstabs : list of DataFrame, optinal
        The param is used to create new metadata.
        The item is cross table year from each event. 
        If there is metadata local, do not need the parameter.
    new : bool, default False
        If True -> even metadata file in local, still create a new one.
        When it is True, crosstabs cannot be None.
        If False -> if file in local, just load.
        
    Returns
    -------
    DataFrame
    """
    original_meta_path = os.path.join(sub_event_path, 'metadata.xlsx')
    processed_meta_path = os.path.join(processed_data_path, 'metadata.xlsx')
    
    if os.path.exists(processed_meta_path) & ~new:
        # load from local
        print('load meta data...')
        df_meta = pd.read_excel(processed_meta_path, index_col=[0, 1])
        return df_meta
    
    print('create meta data...')
    
    # load original df_meta, do not use those number columns
    print('1. load data...')
    df_meta = pd.read_excel(original_meta_path)
    
    # fill out the missing 40 region info
    # get the 40 records
    if df_meta[df_meta.level=='uncertain'].shape[0] > 0:
        print('fill out the missing region info')

        df_region = pd.read_excel(os.path.join(attri_path, 'companies_without_region.xlsx'))

        for i, row in df_region.iterrows():
            # get region values
            level = row['level']
            region = row['region']
            province = row['province']

            # fill out value
            row_index = df_meta[df_meta[company_name]==row['company_name']].index[0]

            df_meta.at[row_index, 'level'] = level
            df_meta.at[row_index, 'region'] = region
            df_meta.at[row_index, 'province'] = province

        # save meta
        df_meta.to_excel(original_meta_path, index=False, freeze_panes=(1, 3))
    
    df_meta = df_meta[[
        'ID', company_name,
        'level', 'region', 'province', 'M_D', # TODO: daughter corresponding
        '企业类型 Corporate type',
        '成立日期 Date of Foundation', 'foundation_year',
        'red', 'black', 'green', 'grey',
        'Latest evaluation year for A taxpayer'
    ]].rename(columns={
        company_name: 'company_name',
        '成立日期 Date of Foundation': 'foundation_date',
        '企业类型 Corporate type': 'corporate_type'
    }).set_index(['ID', 'company_name'])
    
    # merge crosstabs
    print('2. merge with crosstabs...')
    df_crosstabs = pd.concat(crosstabs, axis=1)
    
    # add margins
    df_crosstabs = add_total_event_numbers(df_crosstabs)
    
    # merge meta and events stat crosstabs
    df_meta = df_meta.merge(df_crosstabs, how='left', on=['ID', 'company_name'])
    
    print('3. save metadata...')
    df_meta.to_excel(processed_meta_path, freeze_panes=(1, 2))
    
    return df_meta    
    


# In[50]:


def get_meta_period_table(metadata, crosstabs, date_type, new):
    """
    Merge crosstabs(year or day level), then label, save to local as well as the credit category simplier version.
    The code is extracted from the common code in functions get_meta_year_table() and get_meta_day_table().
    
    Parameters
    ----------
    metadata : DataFrame
        The main metadata which contains company_name, ID, and foundation_year.
    crosstabs : list of DataFrame
    date_type : str
        Can be 'year' or 'day'
    new : bool
        
    Returns
    -------
    DataFrame
    """
    meta_file_name = 'panel_data_' + date_type + '.xlsx'
    flag_file_name = 'panel_data_' + date_type + '_simple.xlsx'
    demo_file_name = 'panel_data_' + date_type + '_demo.xlsx'
    demo_flag_file_name = 'panel_data_' + date_type + '_simple_demo.xlsx'
    
    processed_meta_path = os.path.join(processed_data_path, meta_file_name)
    
    if os.path.exists(processed_meta_path) & ~new:
        # load from local
        df_meta = pd.read_excel(processed_meta_path, index_col=[0, 1, 2])
        return df_meta
    
    print('create meta data ')
    
    # merge crosstabs
    print('1. merge crosstabs...')
    df_meta = pd.concat(crosstabs, axis=1)
    
    # add margins 
    df_meta = add_total_event_numbers(df_meta)
    
    # for day panel, remove the date after e_date
    if date_type == 'day':
        df_meta = df_meta.drop(df_meta[df_meta.index.get_level_values('date').str[:4].astype(int) > e_year].index)
        
        # sort
        df_meta = df_meta.sort_index(level=['ID', 'date'])
        
        # fill the empty dates
        df_meta = df_meta.groupby(level='ID', as_index=False).ffill()
    
    # label red black 
    print('2. add credit rating columns...')
    df_meta = label_companies(df_meta, metadata, date_type)

    df_meta = df_meta.pipe(move_last_column_to_first).pipe(move_last_column_to_first).pipe(move_last_column_to_first)

    # handle missing years in the middle
    if date_type == 'year':
        df_meta = complete_missing_years(df_meta)
    
    # save
    print('3. save...')
    
    # save demo, cuz too big to browse normally
    df_meta[:500].to_excel(os.path.join(processed_data_path, demo_file_name), freeze_panes=(1, 3))
    
    #if date_type == "year":
    # generate flag
    df_flag = df_meta[['green', 'red', 'black', 'redlist', 'blacklist', 'watchlist', 'permit', 'penalty', 'commit']]
    
    # save flag
    df_flag[:500].to_excel(
        os.path.join(processed_data_path, demo_flag_file_name), freeze_panes=(1, 3))
    
    # save the master table
    if date_type == 'day': # too big to save in excel
        flag_file_name = 'panel_data_' + date_type + '_simple.dta'
        
        df_flag.to_stata(
            os.path.join(processed_data_path, flag_file_name), version=118)
        
        meta_file_name = 'panel_data_' + date_type + '.dta'
        
        df_meta.to_stata(os.path.join(processed_data_path, meta_file_name), version=118)
        
        return df_meta
    
    # save normally
    df_flag.to_excel(
        os.path.join(processed_data_path, flag_file_name), freeze_panes=(1, 3))

    df_meta.to_excel(processed_meta_path, freeze_panes=(1, 3))

    return df_meta


# In[11]:


def get_meta_year_table(metadata, crosstabs=None, new=False):
    """
    Load or Create panel_data_year by merge crosstabs_year from events(with red black labels).
    
    Parameters
    ----------
    metadata : DataFrame
        The main metadata which contains company_name, ID, and foundation_year.
    crosstabs : list of DataFrame, optinal
        The param is used to create new metadata.
        The item is cross table year from each event. 
        If there is metadata local, do not need the parameter.
    new : bool
        If True -> even metadata file in local, still create a new one.
        When it is True, crosstabs cannot be None.
        If False -> if file in local, just load.
        
    Returns
    -------
    DataFrame
    """      
    print('handle year-level metadata...')
    
    df_meta_year = get_meta_period_table(
        metadata,
        crosstabs, 
        'year',
        new)
    
    return df_meta_year


# In[12]:


def complete_missing_years(df_meta_year):
    """
    Somehow some companies' years are incomplete.
    
    Parameters
    ----------
    df_meta_year : DataFrame
        
    Returns
    -------
    DataFrame
    """    
    def complete_year(group):
        # generate the right years
        first_year = group.year.tolist()[0]
        last_year = group.year.tolist()[-1]

        consecutive_years = set(range(first_year, last_year+1))

        actual_years = set(group.year)

        if len(consecutive_years) == len(actual_years):
            return group

        missing_years = list(consecutive_years.difference(actual_years))

        # add missing years
        for year in missing_years:
            group = group.append([{'year': year}], ignore_index=True)

        # order and fillna
        return group.sort_values('year').ffill()

    return df_meta_year.reset_index().groupby(
        ['ID']
    ).apply(complete_year).set_index(['ID', 'company_name', 'year'])
    


# In[13]:


def get_meta_day_table(metadata, crosstabs=None, new=False):
    """
    Get panel_data_day by merge crosstabs_day from events.
    
    Parameters
    ----------
    metadata : DataFrame
        The main metadata which contains company_name, ID, and foundation_year.
    crosstabs : list of DataFrame, optinal
        The param is used to create new metadata.
        If there is metadata local, do not need the parameter.
    new : bool
        If True -> even metadata file in local, still create a new one.
        When it is True, crosstabs cannot be None.
        If False -> if file in local, just load.
        
    Returns
    -------
    DataFrame
    """
    print('handle day-level metadata...')
    
    df_meta_day = get_meta_period_table(
            metadata,
            crosstabs, 
            'day',
            new)

    return df_meta_day


# In[14]:


def add_M_D_column(df, df_meta):
    """
    Add M_D column from df_meta to df.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame needs to be added M_D column.
    df_meta : DataFrame
        The metadata with M_D column. 
    
    Returns
    -------
    DataFrame
        A DataFrame with added M_D column.
    """ 
    index_names = df.index.names
    has_multi_index = 'company_name' in index_names
    
    df_left = df.copy()
    
    if has_multi_index:
        df_left = df_left.reset_index()
        
    df_left = df_left.merge(
        df_meta.reset_index()[['company_name', 'M_D']], 
        how='left',
        on='company_name')
    
    if has_multi_index:
        df_left = df_left.set_index(index_names)
        
    return df_left


# In[15]:


def add_M_D_column_for_list(df_list, df_meta):
    """
    Add M_D column from df_meta to a list of DataFrame.
    
    Parameters
    ----------
    df_list : list of DataFrame
        The DataFrame needs to be added M_D column.
    df_meta : DataFrame
        The metadata with M_D column. 
    
    Returns
    -------
    list of DataFrame
    """ 
    return [df.pipe(add_M_D_column, df_meta) for df in df_list]


# # Credit category variations

# In[16]:


def add_credit_category_column(df_meta):
    """
    Combine credit category to one color column and Return a DataFrame with added new color column.
    
    Parameters
    ----------
    df_meta : DataFrame
        The DataFrame with at least columns: company_name, green, red, and black.
        
    Returns
    -------
    DataFrame
    """ 
    # combine green red black to 1 credit rating columns
    df_meta['color'] = df_meta.apply(
        lambda row: 'green' if row['green'] > 0 else ('red' if row['red'] > 0 else 'black')
        , axis=1
    )

    return df_meta


# In[17]:


def check_rating_variation_types(df_meta_year=None):
    """
    Output what kinds of credit category variation types the dataset have.
    
    Parameters
    ----------
    df_meta_year : DataFrame, default None
        The panel_data_year(_simple) DataFrame. If it is None, import from local.
    """   
    if df_meta_year is None:
        df_meta_year = get_credit_panel_data()

    df_rating_change = add_credit_category_column(df_meta_year).reset_index()[['company_name', 'color']]

    # remove the consecutive duplicates
    df_rating_change = df_rating_change.loc[df_rating_change.color.shift() != df_rating_change.color]

    # remove the companies without variation (only one record)
    df_rating_change = df_rating_change[df_rating_change.company_name.duplicated(keep=False)]

    # new column to capture the variation
    df_rating_change['rating_change'] = df_rating_change.groupby('company_name')['color'].transform(lambda x: '->'.join(x))

    print(df_rating_change.head(10))

    print('The dataset exists the following credit rating changing types.')
    print(df_rating_change.rating_change.unique())
    # out: ['green->red' 'green->black' 'green->red->black' 'red->black']
    


# In[18]:


def get_markov_data(df_meta_year):
    """
    Output the percentage of the state transition of credit category.
    
    Parameters
    ----------
    df_meta_year : DataFrame, default None
        The panel_data_year(_simple) DataFrame. If it is None, import from local.
    """  
    if df_meta_year is None:
        df_meta_year = get_credit_panel_data()

    df_color = add_credit_category_column(df_meta_year).reset_index()[['company_name', 'color']]
    
    # add next color
    df_color['color_next'] = df_color.color.shift(-1)
    
    # remove last row in each company group (cuz no need)
    df_color = df_color.groupby(
        'company_name', as_index=False
    ).apply(
        lambda group: group.iloc[:-1]
    ).reset_index(drop=True)
    
    # add color transition column 
    df_color['transition'] = df_color.color + '->' +  df_color.color_next
    
    # show counts, probability can be calculated self
    print(df_color.transition.value_counts().sort_index().to_frame())


# In[19]:


def get_rating_variation_table(df_meta_year):
    """
    Return the credit category variation data by year, which provide data for credit category change chart.
    
    If a separate stat of mother and daughter is needed, should import df_meta_year with only mother or daughter companies.
    
    Parameters
    ----------
    df_meta_year : DataFrame
        The panel_data_year(_simple) DataFrame.
    
    Returns
    -------
    DataFrame
        The DataFrame contains green red black firms and their percentages by year.
    """ 
    # rating variations
    # color by year
    df_rating_change = df_meta_year.reset_index().groupby('year')[['green', 'red', 'black']].sum()

    # total firms 
    df_rating_change['firms'] = df_rating_change.red + df_rating_change.black + df_rating_change.green

    # add pct
    get_pct = lambda color: np.round(df_rating_change[color] / df_rating_change.firms * 100, 2)

    df_rating_change['green_pct'] = get_pct('green')
    df_rating_change['red_pct'] = get_pct('red')
    df_rating_change['black_pct'] = get_pct('black')
    
    return df_rating_change


# In[78]:


def create_metadata_year_for_variation_firms(df_meta_year=None):
    """
    Create and save the variation(credit ratings) frims in df_meta_year.
    
    Parameters
    ----------
    df_meta_year : DataFrame, default None
        The panel_data_year(_simple) DataFrame. If it is None, import from local.
    """   
    path = os.path.join(processed_data_path, 'panel_data_year_credit_category_variation.xlsx')
    
    print('create the crosstable for variation firms...')
    
    # get crosstab_year table
    if df_meta_year is None:
        df_meta_year = pd.read_excel(os.path.join(processed_data_path, 'panel_data_year_simple.xlsx'), index_col=[0,1,2])
    
    time_df = df_meta_year.reset_index()

    # create r b g count table
    df_variation = time_df.groupby('company_name', as_index=False)['red', 'black', 'green'].sum()
    df_variation[['red', 'black', 'green']] = df_variation[['red', 'black', 'green']].astype('bool').astype('int')
    
    def get_data_from_mask(mask):
        variation_firms = df_variation[mask].company_name.values
        print(f'company count: {len(variation_firms)}')
        
        if len(variation_firms) == 0:
            return None
        
        return time_df[time_df.company_name.isin(variation_firms)].set_index(['ID', 'company_name', 'year'])
    
    with pd.ExcelWriter(path) as writer:
        print('Type: red -> black') # 1
        df_rb = get_data_from_mask(((df_variation.red + df_variation.black) == 2) & (df_variation.green == 0))
        if df_rb is not None:
            df_rb.to_excel(writer, sheet_name='red_to_black', freeze_panes=(1,3))
        
        print('Type: green -> red -> black') # 661
        df_grb = get_data_from_mask((df_variation.red + df_variation.black + df_variation.green) == 3)
        df_grb.to_excel(writer, sheet_name='green_red_black', freeze_panes=(1,3))
        
        print('Type: green -> black') # 2302
        df_gb = get_data_from_mask((df_variation.black + df_variation.green) == 2)
        df_gb.to_excel(writer, sheet_name='green_black', freeze_panes=(1,3))
        
        print('Type: green -> red') # 54213
        df_gr = get_data_from_mask(((df_variation.red + df_variation.green) == 2) & (df_variation.black == 0))
        df_gr.to_excel(writer, sheet_name='green_red', freeze_panes=(1,3))


# # Internal consistency check 

# In[21]:


def internal_consistency_check_year(df_meta, df_meta_year):
    """
    Internal consistency check: if the firm's fo
    
    undation year is equal to the earliest year in panel_data_year.
    
    Parameters
    ----------
    df_meta : DataFrame
        The metadata DataFrame.
    df_meta_year : DataFrame
        The panel_data_year(_simple) DataFrame.
    
    Returns
    -------
    DataFrame
        The DataFrame contains green red black firms and their percentages by year.
    """ 
    # internal consistent check
    # 1. check foundation year and the earliest year in df_meta_year
    # get the earliest year
    df_check = df_meta_year.reset_index().groupby('company_name', as_index=False)['year'].min() 

    # join foundation year
    df_check = df_check.merge(
        df_meta.reset_index()[['company_name', 'foundation_year']],
        how='left',
        on='company_name'
        )

    # add match flag
    df_check['flag'] = df_check.year == df_check.foundation_year

    # handle 2014
    df_check.loc[(df_check.year >= df_check.foundation_year) & (df_check.year == 2014), 'flag'] = True
    
    return df_check[df_check.flag==False]


# In[22]:


def get_incorrect_event_idx(df_meta, df_event, start_col='start_date', foundation_col='foundation_date'):
    """
    Internal consistency check: return the record idx where the start year of event record is ealier than the firm's foundation year.
    
    Parameters
    ----------
    df_meta : DataFrame
        The metadata DataFrame.
    df_event : DataFrame
        The processed event DataFrame.
    start_col : str, default 'start_date'
        The column name of start year or date of a event.
    foundation_col : str, default 'foundation_date'
        Can be 'foundation_year' (for redlists)
    
    Returns
    -------
    Int64Index
    """ 
    # join foundation year
    df_check = df_event[['company_name', start_col]].merge(
        df_meta.reset_index()[['company_name', 'foundation_year', 'foundation_date']],
        how='left',
        on='company_name'
        ) # company_name, start_year/date, foundation_year, foundation_date

    # add match flag
    df_check['flag'] = (df_check[start_col] >= df_check[foundation_col])
    
    # ignore those companies without foundation year
    df_check.loc[df_check[foundation_col].isnull(), 'flag'] = True
    
    df_incorrect = df_event[df_check.flag==False]
    
    print(f'{df_incorrect.shape[0]} incorrect records')
    
    return df_incorrect.index


# In[23]:


def remove_incorrect_event_records(df_meta):
    """
    Remove all events' incorect records where the start year of event record is ealier than the firm's foundation year, 
    then Save to local(clean_sub_events).
    
    Parameters
    ----------
    df_meta : DataFrame
        The metadata DataFrame with foundation_year and foundation_date columns.
    """ 
    # prepare events, other sub-events have been verified and do not have incorrect records
    l_path = [
        '11_event_permit.xlsx',
        '22_event_a_taxpayer.xlsx',
        '31_event_dishonest_person.xlsx',
        '4_event_abnormal_operations.xlsx',    
        '51_event_commitment_implementation.xlsx'    
    ]
    
    l_start = [
        'start_year',
        '评价年度 Evaluation year',
        'issue_date',
        'start_date',
        'start_date'
    ]
    
    l_foundation = [
        'foundation_year',
        'foundation_year',
        'foundation_date',
        'foundation_date',
        'foundation_date'
    ]
    
    for path, start_col, foundation_col in zip(l_path, l_start, l_foundation):
        print(f'handle {path}...')
        
        # load event
        save_path = os.path.join(sub_event_path, path)
        df_event = pd.read_excel(save_path, dtype={start_col: str})
        
        # get idx for incorrect records
        idx=get_incorrect_event_idx(df_meta, df_event, start_col, foundation_col)
        
        if path == '11_event_permit.xlsx':
            # for permit, some permits are surely earlier before foundation date
            df_incorrect = df_event[df_event.index.isin(idx)]
            
            # refine the index, keep permits for bank account opening and Enterprise name pre-approval
            idx = df_incorrect[~df_incorrect['许可内容 Permission Content'].str.contains(r'名称.*核准|开户|开立', regex=True)].index
        
        if len(idx) > 0:
            print(f'remove incorrect records: {len(idx)}, then save to local')
            df_without_incorrect = df_event[~df_event.index.isin(idx)]
            
            # save correct records
            df_without_incorrect.to_excel(
                save_path, 
                index=False, 
                freeze_panes=(1, 2))
            


# In[24]:


def internal_consistency_check_credit_category(df_meta, df_meta_period, date_type='year'):
    """
    Create and save two unmatch tables(red and black) including actual and predicted credit categories of 2022.
    
    Parameters
    ----------
    df_meta : DataFrame
        The meta data contains true labels(red and black)
    df_meta_period : DataFrame
    date_type : str, default 'year'
        The value can be 'year' or 'day'.
    """
    # 1. generate df_compare
    print(f'compare {date_type}-level predictions and actual red/black labels...')
    
    df_flag = df_meta_period[['red', 'black', 'redlist', 'blacklist', 'watchlist']]

    # get predicted records
    # only get the last record of each company
    if date_type == 'day':
        df_compare = df_flag.groupby(level='company_name').last()
    else:
        # if handle like 'day', there are some negative errors
        df_compare = df_flag.reset_index()
        df_compare = df_compare[df_compare.year == e_year].set_index(['company_name'])

    # add actual red black by crossing merge with df_meta
    df_compare = df_compare.merge(df_meta[['red', 'black']],
                                 how='left',
                                 on='company_name',
                                 suffixes=('_predicted', '_actual'))
    
    # add flag to show each company is match or not
    df_compare['red_match'] = (df_compare['red_actual'] == df_compare['red_predicted'])
    df_compare['black_match'] = (df_compare['black_actual'] == df_compare['black_predicted'])

    df_compare = df_compare[[
        'red_actual', 'red_predicted', 'red_match',
        'black_actual', 'black_predicted', 'black_match',
        'redlist', 'blacklist', 'watchlist'
    ]]
    
    # 2. Display unmatch records
    # show numbers
    print(f'actual red companies: {df_compare.red_actual.sum()}')
    print(f'actual black companies: {df_compare.black_actual.sum()}')
    
    print(f'predicted red companies: {df_compare.red_predicted.sum()}')
    print(f'predicted black companies: {df_compare.black_predicted.sum()}')

    # red: unmatch
    if (~df_compare.red_match).sum() > 0:
        print('red: unmatch, actual 1 predicted 0')
        df_incorrect = df_compare[(df_compare['red_actual'] == 1) & (df_compare['red_predicted'] == 0)]
        
        print(df_incorrect.shape)
        print(df_incorrect)

        print('red: unmatch, actual 0 predicted 1(should definatily emply)')
        print(df_compare[(df_compare['red_predicted'] == 1) & (df_compare['red_actual'] == 0)])
    else:
        print('Red: all match')

    # black: unmatch
    if (~df_compare.black_match).sum() > 0:
        print('black: unmatch, actual 1 predicted 0')
        print(df_compare[(df_compare['black_actual'] == 1) & (df_compare['black_predicted'] == 0)])

        print('black: unmatch, actual 0 predicted 1(should definatily emply)')
        print(df_compare[(df_compare['black_predicted'] == 1) & (df_compare['black_actual'] == 0)])
    else:
        print('Black: all match')
    
    # 3. SAVE unmatch records
    print('save the unmatch records...')
    with pd.ExcelWriter(os.path.join(processed_data_path, 'unmatch_credit_category_records_' + date_type + '.xlsx')) as writer:
        # save red unmatch
        df_compare[~df_compare.red_match].to_excel(writer, sheet_name='red_unmatch', freeze_panes=(1, 2))
    
        # save black unmatch
        df_compare[~df_compare.black_match].to_excel(writer, sheet_name='black_unmatch', freeze_panes=(1, 2))


# In[25]:


def expand_panel_day_data(df):
    """
    Expand the date in the day panel data to everyday.
    
    Parameters
    ----------
    df : DataFrame
        panel_data_day
    
    Returns
    -------
    DataFrame
    """ 
    df = df.reset_index().copy()
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a date range for each company
    date_ranges = {}
    for company in df['company_name'].unique():
        company_data = df[df['company_name'] == company]
        date_ranges[company] = pd.date_range(start=company_data['date'].min(),
                                            end=company_data['date'].max(),
                                            freq='D')

    # Create an empty DataFrame to store the result
    df_day = pd.DataFrame(columns=['company_name', 'date'])

    # Fill in the rows for each company using their date range
    for company, date_range in date_ranges.items():
        filled_data = pd.DataFrame({'date': date_range, 'company_name': company})
        df_day = pd.concat([df_day, filled_data])


    df_day.date = pd.to_datetime(df_day.date)

    df_day = df_day.merge(
        df, how='left', on=['company_name', 'date']
    ).set_index(
        ['company_name', 'date']
    ).groupby(level='company_name').ffill()
    
    return df_day




# # easy access of data

# In[26]:


def get_credit_panel_data(date_type='year', simple_version=True):
    """
    Load and return panel data of year level or day level/full version or simple version.
    
    Parameters
    ----------
    date_type : str, default 'year'
        Can be 'day'
    simple_version : bool, default True
        Load simple version of panel data if True, otherwise load the original panel data.
    
    Returns
    -------
    DataFrame
    """ 
    # file path
    filename = 'panel_data_' + date_type + ('_simple' if simple_version else '') + ('.xlsx' if date_type == 'year' else '.dta')
    
    path = os.path.join(processed_data_path, filename)
    
    if date_type == 'year':        
        return pd.read_excel(path, index_col=[0, 1, 2])
    else:
        return pd.read_stata(path, index_col=[0, 1, 2]).rename(columns={
            'blacklist_overload_transport_ill': 'blacklist_overload_transport_illegal'
        }) # one column name was shortened when saving, need to recover


# In[27]:


def get_credit_meta_data():
    """
    Load and return credit metadata.
    
    Returns
    -------
    DataFrame
    """ 
    return pd.read_excel(os.path.join(processed_data_path, 'metadata.xlsx'), index_col=[0, 1])


# In[28]:


def get_mother_list():
    """
    Load and return mother table with columns: ID, stock_code, company_name.
    
    Returns
    -------
    DataFrame
    """ 
    path = os.path.join(attri_path, 'mother_stock_codes.xlsx')
    
    return pd.read_excel(path)


# In[29]:


def add_stock_code(df, index_cols):
    """
    Add stock code to dataframe.
    
    Parameters
    ----------
    df : DataFrame
    index_cols : list of str
        The list of index columns of the df, e.g. ['ID', 'company_name', 'day'].
    
    Returns
    -------
    DataFrame
    """ 
    df_stock = get_mother_list()
    
    return df.reset_index().merge(
        df_stock[['company_name', 'stock_code']], 
        on='company_name', 
        how='left'
    ).set_index(index_cols)


# # Integrate performance data

# ## performance credit panel (mothers)

# In[30]:


def get_performance_panel_data(df_meta=None):
    """
    Load or create panel data of mother performance.
    Only mothers have performance data.
    
    Parameters
    ----------
    df_meta : DataFrame, default None
        meta data is used to add foundation_year.
        
    Returns
    -------
    DataFrame
    """ 
    final_perf_path = os.path.join(performance_path, 'mother_performance.xlsx')
    
    if os.path.exists(final_perf_path):
        return pd.read_excel(final_perf_path, index_col=[0, 1, 2])
    
    # not exist -> create performance data
    print('create mother performance data...')
    
    # read original performance table
    df_perf_m = pd.read_stata(os.path.join(original_path, 'mother_performance.dta'))
    df_list_m = pd.read_excel(os.path.join(data_path, 'company_list', 'firm_list.xlsx'))
    
    # keep the columns needed    
    columns_m = ['i', 't', 'name1', 'employee', 
                 'ownership', 'type_ownership', 
                 'IndustryCode', 'IndustryName', 
                 'province_con', 'LargestHolder', 'LargestHolderRate',
                 'IApply', 'IApplyGrant', 
                 'subsidy', 'RD', 
                 'TotalAssets', 'TotalLiability', 'IntangibleAsset', 
                 'ProfitParent', 'NetProfit', 'OperatingEvenue',
                 'OperatingCost', 'OperationProfit']

    columns_rename_m = {
        'i': 'stock_code',
        't': 'year',
        'IApply': 'patents_apply',
        'IApplyGrant': 'patents_grant',
        'ownership': 'organ_type',
        'type_ownership': 'organ_type_code',
        'name1': 'name'
    }

    df_perf = df_perf_m[columns_m].rename(columns=columns_rename_m)
    
    # only keep companies in credit dataset
    # get name-code pairs in credit dataset
    path = os.path.join(attri_path, 'mother_stock_codes.xlsx')
    df_mothers = None
    
    if os.path.exists(path):
        df_mothers = pd.read_excel(path)
    else:    
        # create and save df_mothers (stock_code, company_name)      
        df_company = pd.read_excel(os.path.join(attri_path, 'company_ids.xlsx'))
        
        # mothers in credit dataset
        mothers = list(set(df_list_m.company_name).intersection(set(df_company.company_name))) # 4258 unique company but 4346 stock codes
        print(f'{len(mothers)} mothers')
        
        # add stock codes
        df_mothers = df_list_m[df_list_m.company_name.isin(mothers)][['stock/foldername', 'company_name']].rename(
            columns={'stock/foldername': 'stock_code'}) # DF: stock_code, company_name, use code to filter performance rows
        
        # add ID
        df_mothers = assign_id_to_company(df_mothers)

        df_mothers.to_excel(path, index=False)
    
    # handle rows: 1. only keep companyies in df_mothers, 2. year from 2007
    perf_start_year = 2007
    
    df_perf = df_perf[(df_perf.stock_code.isin(df_mothers.stock_code)) & (df_perf.year >= perf_start_year)]
    
    # unify company name and add ID
    df_perf = df_perf.merge(df_mothers, how='left', on='stock_code').drop(['name'], axis=1)
    #df_perf = df_perf.merge(df_company, how='left', on='company_name')
    
    # add age
    df_perf = df_perf.merge(
        df_meta.reset_index()[['ID', 'foundation_year']],
        on='ID',
        how='left')
    
    print('check every firm has foundation_year: the number of NA in foundation_year')
    print(df_perf.foundation_year.isnull().sum())
    
    df_perf['age'] = df_perf['year'] - df_perf['foundation_year'] + 1
    
    df_perf = df_perf[df_perf.age >= 0] # 4 companies have negative ages due to rename of firms
    
    df_perf = df_perf.drop(['foundation_year'], axis=1)
    
    # sort and set index
    df_perf = df_perf.sort_values(['ID', 'year']).set_index(['ID', 'company_name', 'year'])
    
    # delete empty rows (only year stock_code name value, no other info)
    df_perf = df_perf.replace(
        r'^\s*$', 
        np.nan, 
        regex=True
    ).dropna(
        subset=df_perf.columns[1:], 
        how='all') # columns[0] is stock_code which is always not NAN
    
    # handle NAN in organ_type and industry
    def fillna_in_group(group):
        group[['organ_type', 'organ_type_code',
               'IndustryCode', 'IndustryName'
              ]] = group[['organ_type', 'organ_type_code',
                          'IndustryCode', 'IndustryName'
                         ]].ffill().bfill()

        return group

    df_perf = df_perf.groupby(
            level=[0,1]
        ).apply(fillna_in_group)

    # save and return
    print('save performance data...')
    df_perf.to_excel(final_perf_path, freeze_panes=(1, 3))
    
    return df_perf
    
    


# In[31]:


def get_performance_credit_panel_data(df_perf=None, df_meta_year=None):
    """
    Load or create performance_credit_panel_data for mothers. 
    The function combine mothers' credit panel data and performance panel data
    
    Parameters
    ----------
    df_perf : DataFrame, default None
        Performance panel data of mothers.
    df_meta_year : DataFrame, default None
        Credit panel data of mothers.
    
    Returns
    -------
    DataFrame
    """ 
    final_panel_path = os.path.join(performance_path, 'performance_credit_panel_data.xlsx')
    
    if os.path.exists(final_panel_path):
        return pd.read_excel(final_panel_path, index_col=[0, 1, 2])
    
    # load 2 panel data
    if df_perf is None:
        df_perf = get_performance_panel_data()
    
    if df_meta_year is None:
        df_meta_year = get_credit_panel_data()
    
    # filter out mothers from df_meta_year
    df_credit_mother = df_meta_year[
        df_meta_year.index.get_level_values('ID').isin(
            df_perf.index.get_level_values('ID').unique()
        )
    ]
    
    # combine
    df_combined_panel = df_credit_mother.join(df_perf, how='outer')
    
    print('save the combined panel data...')
    df_combined_panel.to_excel(final_panel_path, freeze_panes=(1, 3))
    
    print('save the combined panel data but only from 2014...')
    df_combined_panel[
        df_combined_panel.index.get_level_values('year') >= (s_year - 1)
    ].to_excel(
        os.path.join(performance_path, 'performance_credit_panel_data_2014.xlsx'), 
        freeze_panes=(1, 3))
    
    return df_combined_panel


# In[32]:


def get_performance_credit_panel_data_2014():
    """
    Load performance_credit_panel_data_2014 for mothers. 
    
    Returns
    -------
    DataFrame
    """ 
    final_panel_path = os.path.join(performance_path, 'performance_credit_panel_data_2014.xlsx')
    
    return pd.read_excel(final_panel_path, index_col=[0, 1, 2])


# ## intergrate daguhters

# In[33]:


def get_mother_daughter_relation_panel_data(df_meta=None, df_meta_year=None):
    """
    Load or create mother_daughter_relation_panel_data. 
    The panel data is a base table including mother and daughter correspondences over year, as well as daughters' credit info.
    
    Parameters
    ----------
    df_meta : DataFrame, default None
    df_meta_year : DataFrame, default None
        Credit panel data(simple version).
    
    Returns
    -------
    DataFrame
    """ 
    path = os.path.join(performance_path, 'mother_daughter_relation_panel_data.xlsx')
    
    if os.path.exists(path):
        print('load relation panel...')
        return pd.read_excel(path, index_col=[0, 1, 2]).reset_index() # index just for decrease file size
    
    print('create relation panel...')
    
    # prepare tables
    df_perf_d = pd.read_stata(os.path.join(original_path, 'daughter_performance.dta'))
    
    if df_meta is None:
        df_meta = get_credit_meta_data()
        
    if df_meta_year is None:
        df_meta_year = get_credit_panel_data()
    
    df_mothers = get_mother_list()
    
    # clean columns & handle stock code (6 digit str->int)
    columns_needed = ['i', 't', 'RalatedParty', 'Relationship']

    columns_rename = {
        'i': 'mother_stock_code',
        't': 'year',
        'RalatedParty': 'daughter_name',
        'Relationship': 'relationship'
    }

    df_relation_panel = df_perf_d[columns_needed].rename(columns=columns_rename).astype({'mother_stock_code': int})
    
    df_relation_panel.relationship = df_relation_panel.relationship.replace({
        '上市公司的子公司': 'Subsidiary',
        '上市公司的联营企业': 'Affiliate',
        '上市公司的合营企业': 'Joint Venture'
    })
    
    # remove doplicated 25102
    df_relation_panel = df_relation_panel[~df_relation_panel.duplicated()]
    
    # add mother ID and name, filter out the mothers not in credit dataset
    df_relation_panel = df_relation_panel.merge(
        df_mothers, how='inner', left_on='mother_stock_code', right_on='stock_code'
    ).rename(columns={
        'company_name': 'mother_name',
        'ID': 'mother_ID'
    })
    
    # only observatons after 2014
    df_relation_panel = df_relation_panel[df_relation_panel.year >= s_year]

    # filter out daughters not in credit dataset
    df_company = pd.read_excel(os.path.join(attri_path, 'company_ids.xlsx'))
    
    df_relation_panel = df_relation_panel.merge(
        df_company, how='inner', left_on='daughter_name', right_on='company_name'
    ).rename(columns={
        'ID': 'daughter_ID'
    })
    
    # add daughters' credit category and event numbers
    df_relation_panel = df_relation_panel.merge(
        df_meta_year.reset_index(), 
        how='left', 
        left_on=['daughter_name', 'year'], 
        right_on=['company_name', 'year'])
    
    # remove the observations whose year is earliar than their foundation year
    df_year = df_meta.reset_index()[['ID', 'foundation_year']]
    
    df_relation_panel = df_relation_panel.merge(df_year, how='left', left_on='daughter_ID', right_on='ID') # add foundtion year
    
    df_relation_panel = df_relation_panel[df_relation_panel.year >= df_relation_panel.foundation_year]
    
    # keep columns needed
    columns_order = ['mother_ID', 'mother_stock_code', 'mother_name', 
                     'year',
                     'daughter_ID', 'daughter_name',
                     'relationship',
                     'green', 'red', 'black', 
                     'redlist', 'blacklist', 'watchlist', 'permit', 'penalty', 'commit']
    
    df_relation_panel = df_relation_panel[columns_order]
    
    # order
    df_relation_panel = df_relation_panel.sort_values(['mother_ID', 'daughter_ID', 'year'])

    # save and return
    print('save panel...')
    
    df_relation_panel.set_index(['mother_ID', 'mother_name', 'mother_stock_code']).to_excel(path, freeze_panes=(1, 3))
    
    return df_relation_panel
    


# In[34]:


def get_mother_daughter_stat_panel_data(df_relation_panel=None):
    """
    Load or create relation_stat_panel_data. 
    The panel data is a aggregated panel table based on mother_daughter_relation_panel_data and including
    mothers and their daughters stat info (number of different daughter type and number of their credit categoreis).
    
    Parameters
    ----------
    df_relation_panel : DataFrame, default None
        The panel data created by foundtion get_mother_daughter_relation_panel_data().
    
    Returns
    -------
    DataFrame
    """ 
    path = os.path.join(performance_path, 'relation_stat_panel_data.xlsx')
    
    if os.path.exists(path):
        print('load panel...')
        return pd.read_excel(path, index_col=[0, 1, 2])
    
    print('create panel...')
    
    # dauther numbers
    df_daughter_count = pd.crosstab(index=[df_relation_panel.mother_name, df_relation_panel.year], 
            columns=df_relation_panel.relationship)
    
    df_daughter_count = add_margins(df_daughter_count, 'total')
    
    # credit category numbers
    df_credit_category_count = df_relation_panel.groupby(['mother_name', 'year'])[['green', 'red', 'black']].sum()
    
    # sum categories by company year relation
    df_credit_relation_count = df_relation_panel.groupby(
        ['mother_name', 'year', 'relationship']
    )[['green', 'red', 'black']].sum(
    ).reset_index(
    ).pivot(index=['mother_name', 'year'],
            columns='relationship',
            values=['green', 'red', 'black']
           )
    
    df_credit_relation_count.columns = ['-'.join(col) for col in df_credit_relation_count.columns.values] # flatten multiindex of columns
    
    # combine the 3 DF into one
    df_relation_stat_panel = pd.concat([
        df_daughter_count, 
        df_credit_category_count, 
        df_credit_relation_count
    ], axis=1).reset_index().rename(columns={
        'mother_name': 'company_name'
    })
    
    # add ID and stock_code
    df_relation_stat_panel = df_relation_stat_panel.pipe(
        assign_id_to_company
    ).set_index(['ID', 'company_name', 'year'])
    
    # save and return
    df_relation_stat_panel.to_excel(path, freeze_panes=(1, 3))
    
    return df_relation_stat_panel
    


# In[35]:


def get_performance_credit_relationship_panel_data(df_credit_perf_panel=None, df_relation_stat_panel=None):
    """
    Load or create performance_credit_relationship_panel_data. 
    The panel data inner merge mother_daughter_stat_panel_data to performance_credit_panel_data.
    
    Parameters
    ----------
    df_credit_perf_panel : DataFrame, default None
        The panel contains data from 2007
    df_relation_stat_panel : DataFrame, default None
        The panel contains data from 2014
    
    Returns
    -------
    DataFrame
    """ 
    path = os.path.join(performance_path, 'credit_performance_relation_panel_data.xlsx')
    
    if os.path.exists(path):
        print('load panel...')
        return pd.read_excel(path, index_col=[0, 1, 2])
    
    print('create panel...')
    
    # rename credit category in df_relation_stat_panel
    df_relation_stat_panel = df_relation_stat_panel.rename(columns={
        'green': 'green_daughters',
        'red': 'red_daughters',
        'black': 'black_daughters',
        'total': 'total_daughters'
    })
    
    # merge
    df_final_panel = df_relation_stat_panel.join(df_combined_panel, how='outer')
    
    # refine year range
    df_final_panel = df_final_panel[
        (df_final_panel.index.get_level_values('year') >= s_year) &
        (df_final_panel.index.get_level_values('year') < 2021)]
    
    # TODO: if the rows without relation or perf need to be deleted
    
    # save and return
    df_final_panel.to_excel(path, freeze_panes=(1, 3))
    
    return df_final_panel
    
    


# In[36]:


# code for generating performance data

"""
df_meta = get_credit_meta_data()
df_meta_year = get_credit_panel_data()

df_performance_panel = get_performance_panel_data(df_meta)

df_combined_panel = get_performance_credit_panel_data(df_performance_panel, df_meta_year) # for regression

df_relation_panel = get_mother_daughter_relation_panel_data(df_meta, df_meta_year)

df_relation_stat_panel = get_mother_daughter_stat_panel_data(df_relation_panel)

df_final_panel = get_performance_credit_relationship_panel_data(df_combined_panel, df_relation_stat_panel)
"""


# In[ ]:




