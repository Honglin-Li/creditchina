#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
event1_pemmit_processing

Processing event permit.
"""

# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# df_event = pd.read_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'))

# # manual check those rows without region info
# t = extract_authorites(df_event['许可机关 Permission Authority']).join(df_event['许可机关 Permission Authority'])
# 
# # show the rows without region, to identify posibble region dict
# s_auth = t.loc[t.level=='uncertain', '许可机关 Permission Authority'].value_counts()
# 
# # check 50 one time
# s_auth[:50]

# In[2]:


def add_regions(df_event):
    """
    Add region related columns to df_event.
    
    This function is implemented by calling :func:`utils.extract_authorites`.
    
    Parameters
    ----------
    df_event : DataFrame
    
    Returns
    -------
    DataFrame
        df_event with region columns.
    """
    # some authorities have no 省 市 区ending, finding the dict by manual checking the rows where level='uncertain'
    replace_dict = {
        '北辰': '天津市北辰开发区',
        '凉山州': '四川省凉山彝族自治州',
        '垫江': '重庆市垫江县',
        '中国人民银行中关村': '北京市海淀区中国人民银行',
        '中国人民银行营业管理部': '北京市西城区中国人民银行营业管理部',
        '沈北': '辽宁省沈阳市沈北新区',
        '黄埔': '广东省广州市黄埔区',
        '杨浦': '上海市杨浦区',
        '鞍钢': '辽宁省鞍钢市',
        '国家能源局东北监管局': '辽宁省沈阳市',
        '黔南州': '贵州省黔南布依族苗族自治州',
        '人民银行武进支行': '江苏省常州市人民银行武进支行',
        '富山工业园': '广东省珠海市斗门区富山工业园',
        '巴南住建委': '重庆市巴南区住建委',
        '阿坝州': '四川省阿坝藏族羌族自治州',
        '海西州': '青海省海西蒙古族藏族自治州',
        '黔西南州': '贵州省黔西南布依族苗族自治州',
        '张浦镇': '江苏省苏州市昆山市张浦镇',
        '嘉善': '浙江省嘉兴市嘉善县',
        '龙岗': '深圳市龙岗区',
        '浦东': '上海市浦东新区',
        '郧阳': '湖北省十堰市郧阳区',
        '甘孜州': '四川省甘孜藏族自治州',
        '顺义': '北京市顺义区',
        '南沙海关': '广东省广州市南沙区南沙海关',
        '中国人民银行高陵支行': '西安市高陵区中国人民银行高陵支行',
        '中国银行保险监督管理委员会合川监管分局': '重庆市合川区中国银行保险监督管理委员会合川监管分局',
        '梅李镇': '江苏省苏州市常熟市梅李镇',
        '黔东南州': '贵州省黔东南苗族侗族自治州',
        '西双版纳州': '云南省西双版纳傣族自治州',
        '西乡': '中国陕西省汉中市西乡县',
        '泰和': '江西省吉安市泰和县',
        '中国人民银行吴江支行': '江苏省苏州市吴江市中国人民银行吴江支行',
        '昌江': '海南省昌江黎族自治县',
        '天穆镇': '天津市北辰区天穆镇',
        '东城': '北京市东城区',
        '贾汪': '江苏省徐州市贾汪区',
        '涪城': '中国四川省绵阳市涪城区',
        '柳北分局': '广西壮族自治区柳州市公安局柳北分局',
        '中国人民银行涪陵中心支行': '重庆市涪陵区中国人民银行涪陵中心支行',
        '红河州': '云南省红河哈尼族彝族自治州',
        '延边州': '吉林省延边朝鲜族自治州',
        '涪城': '四川省绵阳市涪城区',
        '中国人民银行金坛支行': '江苏省常州市金坛区中国人民银行金坛支行',
        '中国人民银行江津中心支行': '重庆市江津区中国人民银行江津中心支行',
        '广陵': '江苏省扬州市广陵区',
        '可克达拉': '新疆维吾尔自治区可克达拉市',
        '泗洪': '江苏省泗洪县',
        '钱塘': '浙江省杭州市钱塘县',
        '香洲': '广东省珠海市香洲区',
        '龙华': '广东省深圳市龙华区',
        '呈贡': '云南省昆明市呈贡区',
        '盛泽镇': '江苏省苏州市盛泽镇',
        '中国人民银行永川中心支行': '重庆市永川区中国人民银行永川中心支行',
        '盛泽镇': '江苏省苏州市盛泽镇',
        '涟水': '江苏省淮安市涟水县',
        '中国人民银行黔江中心支行': '重庆市黔江区中国人民银行黔江中心支行',
        '东钱湖': '中国浙江省宁波市东钱湖',
        '怒江州': '云南省怒江傈僳族自治州',
        '中国人民银行华州支行': '陕西省渭南市华州区中国人民银行华州支行',
        '博州': '新疆维吾尔自治区博尔塔拉蒙古自治州',
        '中国人民银行临潼支行': '西安市临潼区中国人民银行临潼支行',
        '阿拉善': '内蒙古自治区阿拉善盟',
        '猇亭': '湖北省宜昌市猇亭区',
        '大柴旦': '青海省海西蒙古族藏族自治州大柴旦行政区',
        '灵溪': '浙江省温州市灵溪镇',
        '中国银行保险监督管理委员会江津监管分局': '重庆市江津区中国银行保险监督管理委员会江津监管分局',
        '涪陵': '重庆市涪陵区',
        '南沙': '广东省广州市南沙区',
        '邹平': '山东省滨州市邹平市',
        '伊犁州': '新疆维吾尔自治区伊犁哈萨克自治州',
        '集美': '福建省厦门市集美区',
        '双河': '新疆维吾尔自治区双河市',
        '大亚湾': '广东省惠州市大亚湾',
        '秀屿': '福建省莆田市秀屿区',
        '鹿邑': '河南省周口市鹿邑县',
        '九华山': '安徽省池州市青阳县九华山',
        '樊城': '湖北省襄阳市樊城区',
        '平谷': '北京市平谷区',
        '赤坎': '东省江门市开平市赤坎镇',
        '萧山': '浙江省杭州市萧山区',
        '花桥镇': '苏省苏州市昆山市花桥镇',
        '西湖': '浙江省杭州市西湖区',
        '德宏州': '云南省德宏傣族景颇族自治州',
        '渝北': '重庆市渝北区',
        '泊里镇': '山东省青岛市黄岛区泊里镇',
        '孙家岔镇': '陕西省榆林市神木市孙家岔镇',
        '乌翠': '黑龙江省伊春市乌翠区',
        '湖坊': '江西省南昌市青山湖区湖坊镇',
        '长兴岛': '上海市崇明区长兴岛',
        '泾河新城': '陕西省西咸新区泾河新城',
        '黄桥镇': '江苏省泰兴市黄桥镇',
        '城厢': '福建省莆田市城厢区',
        '肥东': '安徽省合肥市肥东县',
        '胡杨河': '新疆维吾尔自治区胡杨河市',
        '澄迈': '海南省澄迈县',
        '海安': '江苏省南通市海安市',
        '海州': '江苏省连云港市海州区',
        '临淄': '山东省淄博市临淄区',
        '博白': '广西壮族自治区玉林市博白县',
        '巴州': '新疆维吾尔自治区巴音郭楞蒙古自治州',
        '宝泉岭': '黑龙江省鹤岗市萝北县宝泉岭',
        '沭阳': '江苏省宿迁市沭阳县',
        '锦溪镇': '江苏省昆山市锦溪镇',
        '祝塘镇': '江苏省无锡市江阴市祝塘镇',
        '新绛': '山西省运城市新绛县',
        '分宜': '江西省新余市分宜县',
        '云岩': '贵阳市云岩区',
        '鹿城': '浙江省温州市鹿城区',
        '海北州': '青海省海北藏族自治州',
        '中宁': '宁夏回族自治区中卫市中宁县',
        '尖草坪': '山西省太原市尖草坪区',
        '甘南州': '甘肃省甘南藏族自治州',
        '宁东': '宁夏回族自治区银川市灵武市宁东镇',
        '里水镇': '广东省佛山市南海区里水镇',
        '恭城': '广西壮族自治区桂林市恭城瑶族自治县',
        '柳南': '广西壮族自治区柳州市柳南区',
        '遥观镇': '江苏省常州市遥观镇',
        '横山桥镇': '江苏省常州市武进区横山桥镇',
        '门头沟': '北京市门头沟区',
        '界埠': '江西省新干县界埠乡',
        '綦江': '重庆市綦江区',
        '海虞镇': '苏省苏州市常熟市海虞镇',
        '灵璧': '安徽省宿州市灵璧县',
        '皇桐镇': '海南省临高县皇桐镇',
        '湘西州': '湘西土家族苗族自治州湘西土家族苗族自治州',
        '酉阳': '重庆市酉阳土家族苗族自治县',
        '歇马镇 ': '湖北省襄阳市保康县歇马镇',
        '合川': '重庆市合川区',
        '白水': '中国陕西省渭南市白水县'
    }
    
    s_auth = \
    df_event['许可机关 Permission Authority'].replace(r'\d+', '', regex=True).replace(replace_dict, regex=True)
    
    # get region info
    df_region = extract_authorites(s_auth)

    # append to original df: province, city, area, level
    return df_event.join(df_region) 


# In[3]:


def process_event():
    """
    Add region columns to df_event, then get cross tables.
    
    This func add category columns to df_event, then save processed df_event to local. 
    Then generate, save, and return cross tables of company and categories and cross tables of [company, year] and categoreis.
    
    Returns
    -------
    df_crosstab_event : DataFrame
        Cross tables of company and categories.
    df_crosstab_event_year : DataFrame
        Cross tables of [company, year] and categoreis.
    df_period : DataFrame
        Data from each time point.
    """
    # load event
    print('load data...')
    df_event = pd.read_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'))
    
    # translate permit_type
    permit_type_map = {
        '其他': 'Other',
        '普通': 'General',
        '核准': 'Approval',
        '特许': 'Concessions',
        '登记': 'Registration',
        '认可': 'Endorsement'
    }
    
    df_event.permit_type = df_event.permit_type.replace(permit_type_map)
    
    # clean columns
    df_event['permit_certificate'] = df_event['许可证书名称 Name of Permission Certificate'].replace(
        ['— —','-','空','无','《','》'], np.nan, regex=True)
    
    df_event['end_date'] = df_event['end_date'].replace('— —', np.nan, regex=True)
    
    print('add region...')
    
    df_event = df_event.pipe(add_regions)
    
    # handle unknown provinces
    df_event = standard_province_names(df_event)
    
    # check distribution
    draw_distribution(
        [df_event.start_year, df_event.province, df_event.level, df_event.permit_type],
        ['bar', 'pie', 'pie', 'pie']
    )
    
    print('save processed df_event...')
    
    df_event[['company_name', 
               'theme',
               'province',
               'level',
               'authority',
               'start_year',
               'start_date', 
               'end_year',
               'end_date',
               'permit_type',
               'permit_certificate',
               '行政许可决定文书名称 Name of Administrative Permission Decision'
              ]].rename(
                    columns={'行政许可决定文书名称 Name of Administrative Permission Decision': 'permit_decision_name'}
                ).to_excel(os.path.join(processed_event_path, 'event1_permit.xlsx'), 
                          index=False, 
                          freeze_panes=(1, 1))

    
    print('generate cross tables of campany and categories...')

    cat_column_names = ['level', 'theme', 'permit_type']
    cat_prefix = ['E1_Permit_Auth_level:', 'E1_Permit_Theme:', 'E1_Permit_Type:']
    number_column_name = 'permit'
    multi_flags = [False, False, False]
    save_prefix = 'event1_permit'
    start_year_col = 'start_year'
    start_date_col = 'start_date'
    end_year_col = 'end_year'
    end_date_col = 'end_date'

    return create_all_crosstabs(df_event,
                                cat_prefix, 
                                cat_column_names, 
                                number_column_name, 
                                multi_flags,
                                save_prefix,
                                start_year_col,
                                start_date_col,
                                end_year_col,
                                end_date_col
                                )
    
    


# process_event()
