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
event_parser_v2

The Module extract event data from credit reports.
"""

from setup import *


# -

# # PDF to list of DataFrame to list of Event String
#
# - extract_event_content(path_to_pdf, n_pages)： pdf to list of DataFrame
# - every DataFrame shape is 1*1, the only value is a string. all the information in one box is extracted into one string
# - every DataFrame = box = Event

def get_event_list_from_pdf(path_to_pdf, n_pages, pdf_event_dict = {}, start=2):
    """
    Extract text from the pdf and split events into a list.
    
    Parameters
    ----------
    path_to_pdf : str
    n_pages : int
        The number of the pages of the pdf.
    pdf_event_dict : dict
    start : int, default 2
        The page to start to extract in the pdf.
        
    Returns
    ----------
    event_string : str
    event_list : list of str
    """
    if path_to_pdf in pdf_event_dict:
        return pdf_event_dict[path_to_pdf]
    else:
        # get list of DF
        list_from_read_pdf = read_pdf(path_to_pdf, pages='3-'+str(n_pages), lattice=True, pandas_options={'header': None}, silent=True)
        
        # get list of event string
        string = list_from_read_pdf[start:] # the fisrt 2 DF are useless
        string_list = [str(i.values[0][0]).replace('行政许可决定书文号', '行政许可决定文书号') for i in string] # list of DF to list of string 
        event_string = ''.join(string_list) # list of string to string. split bar is -------------
        event_list = re.split(r'-{5,}|\|\s{2,}', event_string) # string to Events # r'\r-{5,}\r|\r-{5,}|-{5,}\r|\|   |\|'
        event_list = list(filter(None, event_list))
        
        pdf_event_dict[path_to_pdf] = event_list
        return(event_string, event_list)


# In order to extract the entries of an event in rectangular (table) format, on needs to define a list of entry names (i.e. exact names which define the name of an event item). This allows a structured extraction of the event items as the iten names are recurrent for a given event.

# # Clean Column Name(pattern) and content(Entry)

pattern_dict = {
    # 行政管理 
    1: [
        # 2.1 行政许可
        '行政许可决定文书号:',
        #'行政许可决定书文号:', # old version  new added
        '行政许可决定文书名称:',
        '许可证书名称:',
        '许可类别:',
        '许可编号:',
        '许可决定日期:',
        '许可截止日期:', # new added
        '有效期自:',
        '有效期至:',
        '许可内容:',
        '许可机关:',
        '审核类型:', # new added
        '许可机关统一社会信用代码:',
        '数据来源单位:',
#         '数据来源单位统一社会信用代码:'
        '数据来源单位统一社会信用代',
        '行政处罚决定书文号:',
        # 2.2 行政处罚
        '处罚类别:',
        '处罚决定日期:',
        '处罚内容:',
        '罚款金额(万元):',
        '没收违法所得、没收非法财物', #'没收违法所得、没收非法财物的金额（万元）:'
        '暂扣或吊销证照名称及编号:',
        '违法行为类型:',
        '违法事实:',
        '处罚依据:',
        '处罚机关:',
        '处罚机关统一社会信用代码:',
        '数据来源:',
        '数据来源单位统一社会信用代码:',

                ],

    2: [
        '海关注册编码:',
        "统一社会信用代码:",
        "组织机构代码:",
        "注册地址:",
        "企业资质:", # new added
        "年度:", # new added
        "备注:", # new added
        "首次注册日期:",
        "信用等级:",
        "等级认定时间:",
        "注销标志:",
        "数据来源:",
        
        '纳税人名称:',
        '统一社会信用代码:',
        '纳税人识别号:',
        '评价年度:',
        '数据来源:',
        #交通运输工程建设领域守信典型企业（2020）
        '文件依据:',
        '有效期限:',
       #'企业名称:',
                ],

    3: ['失信被执行人姓名/名称:',
        '身份证号码/组织机构代码:',
        '执行法院:',
        '省份:',
        '执行依据文号:',
        '立案时间:',
        '案号:',
        '做出执行依据单位:',
        '生效法律文书确定的义务:',
        '被执行人的履行情况:',
        '失信被执行人行为具体情形:',
        '发布时间:',
        '已履行部分:',
        '未履行部分:',
        '数据来源:',
        #安全生产领域严重失信惩戒名单
        '单位名称:',
        '统一社会信用代码/工商注册',
        '主要负责人:',
        '注册地址:',
        '失信行为简况:',
        '信息报送机关:',
        '纳入理由:',
        '数据来源:',
        # 税务违法黑名单
        '纳税人名称:',
        '法定代表人或负责人姓名:',
        '统一社会信用代码:',
        '组织机构代码:',
        '纳税人识别号:',
        '案件上报期:',
        '注册地址:',
        '负有直接责任的财务负责人姓',
        '负有直接责任的中介机构信息',
        '案件性质:',
        '主要违法事实:',
        '相关法律依据及税务处理处罚',
        '数据来源:',
        #政府采购严重违法失信行为记录名单
        '机构名称:',
        '统一社会信用代码/组织机构',
        '企业地址:',
        '不良行为的具体情形:',
        '处罚依据:',
        '处罚结果:',
        '记录日期:',
        '登记地点:',
        '处罚截止日期:',
        '数据来源:',
        # 严重违法超限超载
        '入库时间:',
        '车牌号:',
        '营运证号:',
        '道路运输证号:',
        '批次:',
        '失信行为:',
        '数据来源:',
        '企业名称:',
        '法人姓名',
        '法人身份证号:',
    ],

    4: ['企业名称:',
        '统一社会信用代码:',
        '法定代表人:',
        '主体身份代码:',
        '注册号:',
        '列入经营异常名录原因类型名', #'列入经营异常名录原因类型名称:'
        '设立日期:',
        '列入决定机关名称:',
        '数据来源:',
       ],

    5: ['承诺类型:',
        '承诺事由:',
        '做出承诺日期:',
        '承诺受理单位:',
        '承诺履约状态:',
        '数据来源:',
        # new sub-event type 企业信用承诺公示
        '企业名称:', # new added
        '统一社会信用代码:', # new added
        '信用承诺事项:', # new added
        '做出信用承诺时间:', # new added
        '经办人:', # new added
       ],
    
    6: ['企业名称:',
        '统一社会信用代码:',
        '评定内容:',
        '等级:',
        '评定时间:',
        '评定单位:',

       ]
}


def clean_pattern(pattern):
    """
    Remove semicolon and carriage return escape sequence from item names.
    
    Parameters
    ----------
    pattern : str
        
    Returns
    ----------
    str
    """
    clean_pattern = re.sub(':', '', pattern)
    clean_pattern = re.sub('\r', '', clean_pattern)
    return(clean_pattern)


def clean_entry(string):
    """
    Remove noise characters from item content.
    
    Parameters
    ----------
    string : str
        
    Returns
    ----------
    str
    """
    clean_string = re.sub(r'第 \d{1,3} 条', '', string)              # remove count information (which follows the first item of an event entry)
    clean_string = re.sub('\r号:', '', clean_string)            # remove part of item name that moves into item content due to a return line break 
    clean_string = re.sub('\r{0,1}码:', '', clean_string)            # remove part of item name that moves into item content due to a return line break 
    clean_string = re.sub('的金额\\(万元\\):', '', clean_string) # remove part of item name that moves into item content due to a return line break 
    clean_string = re.sub('称:', '', clean_string)              # remove part of item name that moves into item content due to a return line break 
    clean_string = re.sub('\r', '', clean_string)               # remove carriage return escape sequence (\r)
    clean_string = re.sub('名:', '', clean_string)
    clean_string = re.sub('代码:', '', clean_string)
    clean_string = re.sub('号:', '', clean_string)
    clean_string = re.sub('情况:', '', clean_string)
    clean_string = re.sub('及其从业人员信息:', '', clean_string)
    return(clean_string)


# # Extract List of Event dict{pattern: entry}

def item_extractor(string, pattern_start, pattern_stop):
    """
    Extracts content of event item given list of item names.
    
    Parameters
    ----------
    string : str
    pattern_start : int
        The position where the pattern starts
    parttern_stop: int
        The position where the pattern ends
        
    Returns
    ----------
    str
    """
    if string.find(pattern_start) == -1:
        substring='no_match'
        return(substring)
    else:
        start = string.find(pattern_start) + len(pattern_start)
        end = string.find(pattern_stop)
        substring = string[start:end]
        return(substring)


def customize_pattern_list(string, pattern_list):
    """
    Prepare columns for Event
    
    Return pattern list to relevant patterns only sorted by the order it appears in the string.
    
    Parameters
    ----------
    string : str
    pattern_start : int
        The position where the pattern starts
    parttern_stop: int
        The position where the pattern ends
        
    Returns
    ----------
    list of str
    """
    reduced_pattern_list = [pattern for pattern in pattern_list if pattern in string] # the Event string contains those columns
    reduced_pattern_list_sorted = sorted(reduced_pattern_list, key = lambda x: re.search(re.escape(x), string).span()[0]) # sort for extracting content

    return(reduced_pattern_list_sorted)


def moving_window(string, pattern_list, minimum_items):
    """
    Iterate over all item names of an event to extract content for each item.
    
    For one single Event String: Event string to event dict {pattern: entry}
    
    Parameters
    ----------
    string : str
    pattern_list : list of str
    minimum_items: int
        The minimum number of items to match.
        
    Returns
    ----------
    dict
    """
    event_dict = {}
    
    # get existing columns
    pattern_list = customize_pattern_list(string, pattern_list)
    
    # less than minimum_items, it is not the event type
    if len(pattern_list) < minimum_items:
        pass # return empty dict
    
    # Else extract item name and item content from
    else:   
        # Event extaction for first to second-last item:
        for i in range(len(pattern_list)-1): # caution here: last event item will be ignored
            pattern_start = pattern_list[i]
            pattern_stop = pattern_list[i+1]
            event_dict[clean_pattern(pattern_start)] = clean_entry(item_extractor(string, pattern_start, pattern_stop))

        # Event extraction for last item:
        pattern_start = pattern_list[len(pattern_list)-1]
        start = string.find(pattern_start) + len(pattern_start)
        end = len(string)
        substring = string[start:end]
        event_dict[clean_pattern(pattern_start)] = clean_entry(substring)
    
    return(event_dict)


def return_matching_events(event_list, pattern_list, minimum_items):
    """
    Drops events for which a certain minimum number of matched item names is not exceeded.
    
    Moreover, the function extends the remaining events by the complete list of item names (patterns) by adding item names that have not been found with an NA entry.
    For all the Event strings: List of Event string
    
    Parameters
    ----------
    event_list : list of str
    pattern_list : list of str
    minimum_items: int
        The minimum number of items to match.
        
    Returns
    ----------
    dict
    """
    event_dict = [moving_window(event, pattern_list, minimum_items) for event in event_list] # list of event dict      apply pattern extraction (generate item name (keys) - item content (values) dictionary entries)
    event_dict = [event for event in event_dict if len(event)>=minimum_items]         # drop event entries whith less than minimum_items matched
    #event_dict = [event for event in event_dict if 'no_match' not in event.values()]
    return(event_dict)


# Create 'event name - event number' dictionary
event_dict = {1: '行政管理',
              2: '诚实守信',
              3: '严重失信主体名单',
              4: '经营异常',
              5: '信用承诺',
              6: '信用评价',
              7: '司法判决',
              8: '其他信息',
              9: '信用状况提升建议',
             10: '公共信用信息地方补充内容'}

# # Main API for extract Events: create_event_df()

# +
## The reason we have this is because some of properties are too long, so they are written in multi-lines:
rename_columns_event = {
    '数据来源单位统一社会信用代': '数据来源单位统一社会信用代码',
    '没收违法所得、没收非法财物': '没收违法所得、没收非法财物的金额（万元）',
    '列入经营异常名录原因类型名': '列入经营异常名录原因类型名称',
}
def remove_unnecessary_phrases(text):
    """
    Remove the text we do not need.
    
    Parameters
    ----------
    text : str
        The position where the pattern ends
        
    Returns
    ----------
    str
    """
    if type(text) == str:
        text = text.replace('查询期内无相关信息', "")
        text = text.replace('nan', "")
        text = text.replace('建议秉持诚信理念,合法有序开展经营活动。', '')
        text = text.replace('许可有效期:— —', '')
    return text
    
def create_event_df(df, event_number, pattern_list, minimum_items, pdf_event_dict):
    """
    Iterate over all documents creating an event-specific dataframe.
    
    Parameters
    ----------
    df : DataFrame
        The dataframe includes pdf_path and the number of pages for each credit reports to be extracted.
    event_number : int
        Can be 1-6. The type of event to extract.
    parttern_list: list
        The list of how the event attributes are structured.
    minimum_items : int
        The minimum number of event attributes to match.
    pdf_event_dict : dict
        
    Returns
    ----------
    DataFrame
    """
    df_events = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # if no corresponding events in Metadata, move to next
        if int(row[event_dict[event_number]]) == 0:
            pass
        else:
            company_id = index
            n_pages = row.n_pages
            file_path = row.file_path
            
            #list_from_read_pdf = extract_event_content(file_path, n_pages)
            #event_list = events_to_list(list_from_read_pdf, start=2)
            event_list = get_event_list_from_pdf(file_path, n_pages, pdf_event_dict, start=2)
            df_event = pd.DataFrame(return_matching_events(event_list, pattern_list, minimum_items=minimum_items))
            df_event['统一社会信用代码'] = row['统一社会信用代码']
            df_event['event_number'] = event_number
            df_event['file_path'] = file_path
            df_event['company_name'] = row['机构名称']
                
            df_events.append(df_event)
    if len(df_events)==0:
        df_events = pd.DataFrame()
    else:
        df_events = pd.concat(df_events).set_index('统一社会信用代码', drop=True)
        df_events = df_events[ ['event_number'] + [ col for col in df_events.columns if col != 'event_number' ] ] # put event_number to the first column
        df_events = df_events.rename(columns=rename_columns_event) # rename some wrap column names
        df_events[:] = df_events.applymap(remove_unnecessary_phrases)
    return(df_events)


# +
# rename Chinese column to Chinese + English
rename_columns = {
        '行政许可决定文书号': '行政许可决定文书号 Administrative Permission Decision Document Code' ,
        '行政许可决定文书名称': '行政许可决定文书名称 Name of Administrative Permission Decision',
        '许可证书名称': '许可证书名称 Name of Permission Certificate' ,
        '许可类别': '许可类别 Type of Permission',
        '许可编号': '许可编号 Number of Permission',
        '许可决定日期':'许可决定日期 Permission decision date' ,
        '许可截止日期': '许可截止日期 Permission deadline date',
        '有效期自': '有效期自 Permission  Decision Valid From',
        '有效期至': '有效期至 Valid Until',
        '许可内容': '许可内容 Permission Content',
        '许可机关': '许可机关 Permission Authority',
        '审核类型': '审核类型 Audit Type',
        #'许可有效期:': '许可有效期 Permission Validity Period',
        '许可机关统一社会信用代码': '许可机关统一社会信用代码 Permission Authority USCI',
        '数据来源单位': '数据来源单位 Data Sources Unit',
        '行政处罚决定书文号': '行政处罚决定书文号 Administrative Penalty Decision Document Number',
        '处罚类别': '处罚类别 Penalty Type' ,
        '处罚决定日期': '处罚决定日期 Penalty Date',
        '处罚内容': '处罚内容 Penalty Content',
        '罚款金额(万元)': '罚款金额(万元) Fine Amount, 10k Yuan',
        '没收违法所得、没收非法财物的金额（万元）': '没收违法所得、没收非法财物的金额（万元）Confiscation of Illegal Gains, 10k Yuan',
        '暂扣或吊销证照名称及编号':'暂扣或吊销证照名称及编号 Suspension or Revocation of License Name and Number',
        '违法行为类型': '违法行为类型 Type of Illegal Behavior',
        '违法事实': '违法事实 Illegal Facts',
        '处罚依据': '处罚依据 Penalty Basis',
        '处罚机关': '处罚机关 Penalty Enforcement Authority',
        '处罚机关统一社会信用代码': '处罚机关统一社会信用代码 Penalty Enforcement Authority USCI',
        '数据来源': '数据来源 Data sources',
        '数据来源单位统一社会信用代码': '数据来源单位统一社会信用代码 Data Sources Unit USCI',
        '海关注册编码': '海关注册编码 Customs Record Number',
        "统一社会信用代码": '统一社会信用代码 Unified Social Credit Identifier',
        "组织机构代码": '组织机构代码 Organizational institution identifier',
        "注册地址": '注册地址 Registration address',
        "企业资质": '企业资质 Firm Qualification',
        "年度": '年度 Year', 
        "备注": '备注 Comment',
        "首次注册日期": '首次注册日期 First registration date',
        "信用等级": '信用等级 Credit level',
        "等级认定时间": '等级认定时间 Accreditation time',
        "注销标志": '注销标志 Deregistration sign',
        '纳税人名称': '纳税人名称 Taxpayer name',
        '纳税人识别号': '纳税人识别号 Taxpayer ID',
        '评价年度': '评价年度 Evaluation year',
        '失信被执行人姓名/名称': '失信被执行人姓名/名称 Name of The Dishonest Person Subject to Execution',
        '身份证号码/组织机构代码': '身份证号码/组织机构代码 ID Number/Organizational ID',
        '执行法院': '执行法院 Execution Court',
        '省份': '省份  Province',
        '执行依据文号': '执行依据文号 Execution Base Number',
        '立案时间': '立案时间 Date of Filing',
        '案号': '案号 Case Number',
        '做出执行依据单位': '做出执行依据单位 Execution Base Unit',
        '生效法律文书确定的义务': '生效法律文书确定的义务 Obligations Determined by Effective Legal Documents',
        '被执行人的履行情况': '被执行人的履行情况 Implementation Performance of Person of Execution',
        '失信被执行人行为具体情形': '失信被执行人行为具体情形 Person of Execution Untrustworthy Behavior Details',
        '发布时间': '发布时间 Date of Issue',
        '已履行部分': '已履行部分 Part of Accomplishment',
        '未履行部分': '未履行部分 Part of Non-accomplishment',
        '企业名称': '企业名称 Firm name',
        '法定代表人': '法定代表人 Legal representative',
        '主体身份代码': '主体身份证代码 Entity ID code',
        '注册号': '注册号 Registration number',
        '列入经营异常名录原因类型名称': '列入经营异常名录原因类型名称 Reason type for listing in Operational abnormality',
        '设立日期': '设定日期 Establishment date',
        '列入决定机关名称': '列入决定机关名称 Listing decision authority name',
        '承诺类型': '承诺类型 Commitment type',
        '承诺事由': '承诺事由 Commitment reason',
        '做出承诺日期': '做出承诺日期 Commitment date',
        '承诺受理单位': '承诺受理单位 Commitment processing unit',
        '承诺履约状态': '承诺履约状态 Commitment implementation status',
        '信用承诺事项': '信用承诺事项 Credit Commitment Matters',
        '做出信用承诺时间': '做出信用承诺时间 Time to make a credit commitment',
        '经办人': '经办人 Manager',
        '评定内容': '评定内容 Assessment content',
        '等级': '等级 Rank',
        '评定时间': '评定时间 Assessment date',
        '评定单位': '评定单位 Assessment unit',
    #安全生产领域严重失信惩戒名单
        '单位名称':'单位名称 Unit name',
        '统一社会信用代码/工商注册':'统一社会信用代码/工商注册号 Uniform social credit code/business registration number',
        '主要负责人':'主要负责人 Principal',
        '失信行为简况':'失信行为简况 Default behavior profile',
        '信息报送机关':'信息报送机关 Information reporting authority',
        '纳入理由':'纳入理由 Reason for inclusion',
        # 税务违法黑名单
        '法定代表人或负责人姓名':'法定代表人或负责人姓名 Name of legal representative or person in charge',
        '组织机构代码':'组织机构代码 Organization code',
        '纳税人识别号':'纳税人识别号 Taxpayer identification number',
        '案件上报期':'案件上报期 Case reporting period',
        '负有直接责任的财务负责人姓':'负有直接责任的财务负责人姓名 Name of the financial person directly responsible',
        '负有直接责任的中介机构信息':'负有直接责任的中介机构信息及其从业人员信息 Information of intermediaries with direct responsibility and information of their practitioners',
        '案件性质':'案件性质 Nature of the case',
        '主要违法事实':'主要违法事实 Main illegal facts',
        '相关法律依据及税务处理处罚':'相关法律依据及税务处理处罚 Relevant legal basis and tax treatment and punishment',
        #政府采购严重违法失信行为记录名单
        '机构名称':'机构名称 Organization name',
        '统一社会信用代码/组织机构':'统一社会信用代码/组织机构代码 Uniform social credit code/organization code',
        '企业地址':'企业地址 Business address',
        '不良行为的具体情形':'不良行为的具体情形 Specific circumstances of the bad behavior',
        '处罚依据':'处罚依据 Penalty basis',
        '处罚结果':'处罚结果 Penalty result',
        '记录日期':'记录日期 Record date',
        '登记地点':'登记地点 Registration location',
        '处罚截止日期':'处罚截止日期 Penalty deadline',
        '文件依据':'文件依据 Document basis',
        '有效期限':'有效期限 Period of validity',
        #严重违法超限超载运输当事人名单
        '入库时间': '入库时间 Registration time',
        '车牌号': '车牌号 License number',
        '营运证号': '运营证号 Operation certificate number',
        '道路运输证号': '道路运输证号 Road transport certificate number',
        '批次': '批次 Batch',
        '失信行为': '失信行为 Breach of trust',
        '来源': '来源 Source'
}

meta_rename_columns = {
        '统一社会信用代码': '统一社会信用代码 Unified Social Credit Identifier, USCI',
        '机构名称': '机构名称 Institution name',
        '企业类型': '企业类型 Corporate type' ,
        '住所': '住所 Address',
        '法定代表人/负责_x000D_人/执行事务合伙_x000D_人': '法定代表/负责人/执行事务合伙人 Legal representative/Person in charge/Executive business partner',
        '成立日期':'成立日期 Date of Foundation' ,
        '行政管理': '行政管理 Administrative management',
        '严重失信主体名单': '严重失信主体名单 Severe untrustworthy entity list',
        '信用承诺': '信用承诺 Credit commitment',
        '司法判决': '司法判决 Judicial decision',
        '诚实守信': '诚实守信 Honesty and trustworthy',
        '经营异常': '经营异常 Abnormal operation',
        '信用评价': '信用评价 Credit assessment',
        '其他信息': '其他信息 Other information',
    # new
        '法定代表人': '法定代表人 Legal representative',
        '举办单位': '举办单位 Organizers',
        '审批机关': '审批机关 Approving authority',
        '法定代表人姓名': '法定代表人姓名 Name of Legal representative',
        '业务主管单位名称': '业务主管单位名称 Name of business unit',
        '组织类型': '组织类型 Organization type',
        '登记类型': '登记类型 Registration Type',
        '业务范围': '业务范围 Business scope'
    }
# -

# # for test
# df_meta = pd.read_csv('output.csv')
# #pdf_dict = {}
# df_event1 = create_event_df(df_meta[:3], event_number=1, pattern_list=pattern_dict[1], minimum_items = 5, pdf_event_dict=pdf_dict)

# # Testing 

# temp = extract_event_content(r"X:\SCS\Monthers_v2.0_correct_one - Kopie\998.pdf", 14)
# #temp[5].iloc[0,0]
# temp = events_to_list(temp, start=2)


# d = return_matching_events(temp, pattern_dict[1], 3)

# d

# clean_entry(temp[1])

# pattern_list = ['企业名称:',
#                 '统一社会信用代码:',
#                 '法定代表人:',
#                 '主体身份代码:',
#                 '注册号:',
#                 '列入经营异常名录原因类型名',
#                 '设立日期:',
#                 '列入决定机关名称:',
#                 '数据来源:' 
#                 ]

# + active=""
# [customize_pattern_list(string, pattern_list3) for string in temp]

# + active=""
# [moving_window(string, pattern_list3) for string in temp]

# + active=""
# return_matching_events(temp, pattern_list3, minimum_items=6)
# -

# Due to wrong formatting in PDF of 91321084565321813C (page 3/30 bottom) minor error in parsing event 6. Unfortunately, this cannot be solved programmatically.

# Due to wrong formatting in PDF of 91321084565321813C (page 3/10 bottom) minor error in parsing event 2. Unfortunately, this cannot be solved programmatically.

#
