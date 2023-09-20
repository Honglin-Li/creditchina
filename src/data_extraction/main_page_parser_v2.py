# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
"""
main_page_parser_v2

The module extract data from the main page of the credit report.
"""

from setup import *


# -

# path = r'X:/SCS/Monthers_v2.0_correct_one - Kopie/122.pdf'
# parsing_iterator(path)

def extract_title_page(path_to_pdf):
    """
    Extract the data on the cover page(basic info).
    
    Parameters
    ----------
    path_to_pdf : str
        The path to the pdf to be extracted.
        
    Returns
    ----------
    DataFrame
    """
    tables = read_pdf(path_to_pdf, pages=2, lattice=True, pandas_options={'header': None}, silent=True)
    
    if len(tables) == 1: # 杭州明州医院
        tab0 = tables[0]
        tab1 = read_pdf(path_to_pdf, pages=3, lattice=True, pandas_options={'header': None}, silent=True)[1]
    #if len(tables) <= 4: this cannot be a condition
    elif tables[0].shape[1] > 3:
        tab0 = tables[0]
        tab1 = tables[1]
    else:
        tab0 = tables[1] # 0 light. basic info
        tab1 = tables[2] # stat
    
    rows = tab0.shape[0]
    temp0 = pd.concat([tab0.iloc[0:rows,0:2], tab0.iloc[0:rows-1,2:4].rename(columns={2: 0, 3: 1})], axis=0, ignore_index=True).transpose()
    df0 = temp0.rename(columns=temp0.iloc[0]).iloc[1:]
    
    temp1 = pd.concat([tab1.iloc[0:4,0:2], tab1.iloc[0:4,2:4].rename(columns={2: 0, 3: 1})], axis=0, ignore_index=True).transpose()
    df1 = temp1.rename(columns=temp1.iloc[0]).iloc[1:]
    
    df = pd.concat([df0, df1], axis=1)
    return(df)


def extract_traffic_light(path_to_pdf):
    """
    Extract the credit category.
    
    Parameters
    ----------
    path_to_pdf : str
        
    Returns
    ----------
    credit_category : str
    company_name : str
    """
    traffic_dict = {}
    
    temp = extract_text(path_to_pdf, page_numbers=[2]).splitlines()
    temp = [i for i in temp if i != '']
    
    # extract company name:
    try:
        company_name = temp[temp.index('企业名称：') + 1] # hidden text, not show in the pdf
    except:
        company_name = extract_institution_name(path_to_pdf)
        # record the company
        with open(r'X:/SCS/04_Results/disregistered.txt', 'a', encoding='utf-8') as disregiestered:
            disregiestered.write("%s\n" % str(company_name))
    
    # traffic_light
    if '存续' in temp:
        traffic_dict['green'] = 1
    else:
        traffic_dict['green'] = 0
        
    if '守信激励对象' in temp:
        traffic_dict['red'] = 1
    else:
        traffic_dict['red'] = 0
        
    if '失信惩戒对象' in temp:
        traffic_dict['black'] = 1
    else:
        traffic_dict['black'] = 0
        
    if '注销' in temp:
        traffic_dict['grey'] = 1
    else:
        traffic_dict['grey'] = 0
    
    return(traffic_dict, company_name)


def extract_institution_name(path_to_pdf):
    """
    Extract the company name.
    
    Parameters
    ----------
    path_to_pdf : str
        
    Returns
    ----------
    str
    """
    all_text = extract_text(path_to_pdf, page_numbers=[0]).splitlines()
    # remove empty strings
    all_text = [text for text in all_text if text != '']
    
    # name is in between 机构名称 (name) and 统一社会信用代码 (usci)
    index_name = [index for index, text in enumerate(all_text) if '机构名称' in text][0]
    index_usci = [index for index, text in enumerate(all_text) if '统一社会信用代码' in text][0]
    
    # extract the name. join used here because if name is too long, it will be written in more than 1 line.
    name = ''.join(all_text[index_name + 1:index_usci])
    return name


event_dict = ['行政管理',
              '严重失信主体名单',
              '信用承诺',
              '司法判决',
              '诚实守信',
              '经营异常',
              '信用评价',
              '其他']
def parsing_iterator(path_to_pdf):
    """
    Extract data for one company.
    
    Parameters
    ----------
    path_to_pdf : str
        
    Returns
    ----------
    dict
    """
    title_page = extract_title_page(path_to_pdf)
    title_page.set_index('统一社会信用代码', inplace=True)
    
    # extract traffic light
    traffic_light, company_name = extract_traffic_light(path_to_pdf)
    
    #title_page['机构名称'] = extract_institution_name(path_to_pdf)
    title_page['机构名称'] = company_name
    
    # add name and re-order column names
    new_order = ['机构名称']
    new_order.extend([c for c in title_page.columns.values if c != '机构名称'])
    title_page = title_page[new_order]
    
    traffic_light = pd.DataFrame.from_records(traffic_light, index=title_page.index)
       
    # merge them together
    row = pd.concat([title_page, traffic_light], axis=1)
    
    # remove the 条
    for event in event_dict:
        row[event] = pd.Series([int(re.sub(',', '', re.sub('条', '', text))) for text in row[event].values]).values
    
    
    row['event_count'] = row.loc[:,[col for col in row.columns if col in event_dict]].sum(axis=1)
    row['n_pages'] = len(list(extract_pages(path_to_pdf)))
    row['file_path'] = path_to_pdf
    return(row)

# +
# Function which iterates over all documents and returns dataframe with main page information


def create_main_df(pdf_paths):
    """
    Extract data on cover page for list of pdf paths.
    
    Parameters
    ----------
    pdf_paths : list of str
        
    Returns
    ----------
    DataFrmae
    """
    df = pd.concat([parsing_iterator(pdf) for pdf in tqdm(pdf_paths, position=0, leave=True)])
    #Drop duplicate companies if there are any
    df = df.reset_index()#.drop_duplicates(subset='统一社会信用代码').set_index('统一社会信用代码', drop=True)
    
    return(df)
# -

# path_to_pdf = r'X://SCS//Daughters-Duplicated//72226.pdf'
# tables = read_pdf(path_to_pdf, pages=2, lattice=True, pandas_options={'header': None}, silent=True)
# tables[0]

# # special companies
# create_main_df([r'X://SCS//Daughters-Duplicated//72226.pdf', r'X://SCS//Daughters-Duplicated//21057.pdf'])

# extract_title_page(path_to_pdf)


