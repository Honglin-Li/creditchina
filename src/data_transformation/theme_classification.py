#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
theme_classification

The module provides data processing, train and tuning, predictions, evaluations in the theme classifications
"""

# for jupyter notebook
from pathlib import Path
import sys

sys.path.insert(0, str(Path().resolve().parents[0].parents[0]))

from src.data_transformation.utils import *


# In[2]:


# ml packages
import pickle
from wordcloud import WordCloud, STOPWORDS
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, recall_score, precision_score, accuracy_score


# # pre-precessing
# 
# - data source
# - Chinese cut
# - word cloud for each theme

# In[4]:


cn_stopwords = [k.strip() for k in open(os.path.join(data_path, 'cn_stopwords.txt'), encoding='utf8').readlines() if k.strip() != '']

def cut_phrases(x, sep=' '):
    """
    Cut phrases in ideograms groups by jieba package.
    
    Parameters
    ----------
    x : str
        The text to be cut.
    sep : str
        The separator to join the cut words.
    
    Returns
    -------
    str
    """
    return sep.join([word for word in jieba.cut(x, cut_all=False) if (word not in cn_stopwords) & (len(word) > 1)])


# In[46]:


# after manual collection: keywords-theme
# authority-map_to->theme, get content-theme datasource
def prepare_data_source(df_event, content_col, theme_col): 
    """
    Prepare data source for theme classification parallelly.
    
    Parameters
    ----------
    df_event : DataFrame
    content_col : list of str or str
        Each item is the column name of content.
    theme_col : str
        The column name of authority.
    
    Returns
    -------
    ds_train : DataFrame
        The records with themes and content count > 1, contains columns: theme, content, count.
    ds_unseen : DataFrame
        The records without themes to be predicted, contains columns: theme, content, count, index.
    df_train : DataFrame
        Straitifing sampled from ds_train(80%)
    df_test : DataFrame
        Straitifing sampled from ds_train(20%)
    """        
    p = Parallel(n_jobs=cpu_count)
    
    # theme
    df_themes = pd.read_excel(os.path.join(attri_path, 'authority_themes.xlsx'))
    
    # clean authority
    s_clean_authorities = df_event[theme_col].str.replace(r'[0-9a-zA-Z()\s.-]+', '', regex=True)
    
    # find match keywords
    auth_pattern = re.compile('|'.join(df_themes.keyword))
    s_auth_keywords = s_clean_authorities.str.findall(auth_pattern)
    
    # extract theme
    def extract_theme(value):
        keyword = np.nan

        if (value != value) or len(value) == 0: # value != value-----> nan
            return keyword

        keyword = value[0]

        return df_themes.loc[df_themes.keyword == keyword, 'theme_en'].tolist()[0]
    
    #s_theme = s_auth_keywords.apply(extract_theme)
    s_theme = p(delayed(extract_theme)(x) for x in tqdm(s_auth_keywords.values, desc='extract theme'))
    
    # content: 
    """ somehow parallel_apply throw exception when I run outside this .py
    s_content = df_event[content_col].replace(
                            r'[0-9a-zA-Z()\s.-]+', '', regex=True
                        ).parallel_apply(lambda row: '.'.join(row.dropna().astype(str)), axis=1) # remove numbers
    """
    connect_func = lambda row: '.'.join(row.dropna().astype(str))
    
    s_content = p(delayed(connect_func)(row) for i, row in tqdm(df_event[content_col].replace(
                            r'[0-9a-zA-Z()\s.-]+', '', regex=True
                        ).iterrows(), desc='connect content'))
    
    ds = pd.concat({
                    'content':pd.Series(s_content), 
                    'theme': pd.Series(s_theme)
                    }, axis=1)
    
    print(f'{ds.shape[0]} records totally. {ds.theme.isnull().sum()} records have no themes')

    print('theme distribution')
    draw_bar(ds.theme)
    
    # cut phrases & count
    with parallel_backend('threading', n_jobs=cpu_count):
        ds['content'] = Parallel()(delayed(cut_phrases)(x) for x in tqdm(ds.content.values, desc='cut content'))
    
    ds['count'] = ds.content.str.len()
    
    # split data set
    ds_unseen = ds[ds.theme.isnull()].reset_index() # keep tracking the index
    
    ds_train = ds[ds.theme.notnull()].reset_index()
    
    print(f'records without theme: {ds_unseen.shape[0]}; records with theme: {ds_train.shape[0]}')
    
    # process train data
    ds_train = ds_train[ds_train['count'] > 1] # fileter by text counts
    
    print(f'training set after removing the records with empty content: {ds_train.shape[0]}')
    
    # split training set to training and test
    df_train, df_test = train_test_split(
        ds_train[['content', 'theme', 'count']], 
        test_size=0.2, 
        random_state=42, 
        stratify=ds_train['theme'])

    return ds_train, ds_unseen, df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    


# In[15]:


def prepare_permit_ds(df_event_permit=None):
    """
    Return the DataFrame for permit.
    
    Parameters
    ----------
    df_event_permit : DataFrame, default None
        if no value, load from local.
    
    Returns
    -------
    ds_train : DataFrame
    ds_unseen : DataFrame
    df_train : DataFrame
    df_test : DataFrame
    """
    if df_event_permit is None:
        df_event_permit = pd.read_excel(os.path.join(sub_event_path, '11_event_permit.xlsx'))
    
    # prepare_permit_ds() needs to much time to run
    save_dir = os.path.join(data_path, 'models', 'train')
    
    if os.path.exists(os.path.join(save_dir, 'permit_ds_train.xlsx')):
        # load data
        ds_train = pd.read_excel(os.path.join(save_dir, 'permit_ds_train.xlsx'))
        ds_unseen = pd.read_excel(os.path.join(save_dir, 'permit_ds_unseen.xlsx'))
        df_train = pd.read_excel(os.path.join(save_dir, 'permit_df_train.xlsx'))
        df_test = pd.read_excel(os.path.join(save_dir, 'permit_df_test.xlsx'))
        
        return ds_train, ds_unseen, df_train, df_test
    
    # create
    return prepare_data_source(df_event_permit, 
                   ['行政许可决定文书名称 Name of Administrative Permission Decision',
                    '许可证书名称 Name of Permission Certificate',
                    '许可内容 Permission Content'
                    ],
                    '许可机关 Permission Authority'
                   )

def prepare_penalty_ds(df_event_penalty=None):
    """
    Return the DataFrame for penalty.
    
    Parameters
    ----------
    df_event_penalty : DataFrame, default None
        if no value, load from local.
    
    Returns
    -------
    ds_train : DataFrame
    ds_unseen : DataFrame
    df_train : DataFrame
    df_test : DataFrame
    """
    if df_event_penalty is None:
        df_event_penalty = pd.read_excel(os.path.join(sub_event_path, '12_event_penalty.xlsx'))
        
    return prepare_data_source(df_event_penalty, 
                   ['行政处罚决定书文号 Administrative Penalty Decision Document Number',
                    '处罚内容 Penalty Content',
                    '违法行为类型 Type of Illegal Behavior',
                    '违法事实 Illegal Facts',
                    '处罚依据 Penalty Basis'
                    ],
                    '处罚机关 Penalty Enforcement Authority'
                   )

def prepare_commitment_ds(df_event_commitment=None):
    """
    Return the DataFrame for credit commitment.
    
    Parameters
    ----------
    df_event_commitment : DataFrame, default None
        if no value, load from local.
    s_count : int, default 200
    
    Returns
    -------
    ds_train : DataFrame
    ds_unseen : DataFrame
    df_train : DataFrame
    df_test : DataFrame
    """
    if df_event_commitment is None:
        df_event_commitment = pd.read_excel(os.path.join(sub_event_path, '51_event_commitment_implementation.xlsx'))
        
    return prepare_data_source(df_event_commitment, 
                   ['承诺类型 Commitment type',
                    '承诺事由 Commitment reason'
                    ],
                    '承诺受理单位 Commitment processing unit'
                   )


# In[13]:


def get_frequency_table(s_content, word_count=50):
    """
    Get frequency DataFrame from Series s_content with at most word_count words.
    
    Parameters
    ----------
    s_content : Series
        The Series contains the content to extract frequency words.
    word_count: int, default 50
        The value of parameter max_features of CountVectorizer class.
    
    Returns
    -------
    df_freq : DataFrame
        Contains index(word), word_freq, doc_freq columns.
    freq_dict : dict
        The dict {word: doc_freq} is the input of wordcloud.
    """
    # get doc-word matrix
    cv = CountVectorizer(max_features=word_count, stop_words='english')

    X = cv.fit_transform(s_content)
    
    df_count_matrix = pd.DataFrame(X.toarray(), columns = cv.get_feature_names_out())

    # frequency
    s_freq = df_count_matrix.sum()

    s_doc_freq = (df_count_matrix>0).sum()

    df_freq = pd.concat({
                        'word_freq':s_freq, 
                        'doc_freq': s_doc_freq
                        }, axis=1).sort_values('doc_freq', ascending=False)
    
    # prepare the input for wordcloud
    freq_dict = df_freq.doc_freq.to_dict()
    
    return df_freq, freq_dict


# In[12]:


# analysis word list
def show_wordcloud(freq_dict, title = None):
    """
    Show wordcloud.
    
    Parameters
    ----------
    freq_dict : dict
    title : str
    """    
    wordcloud = WordCloud(
        background_color='white',
        font_path=font_path,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    )
    
    wordcloud.generate_from_frequencies(freq_dict)

    fig = plt.figure(1, figsize=(12,12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=40, fontproperties=prop)
        fig.subplots_adjust(top=1.3)

    plt.imshow(wordcloud)
    plt.savefig(os.path.join(attri_path, 'wordclouds\\' + title + '.png'))
    plt.show()
    


# # Classification: Model 

# In[1]:


def get_balanced_training_set(df_train, s_count=200):
    """
    Prepare data source for theme classification parallelly.
    
    Parameters
    ----------
    df_train : DataFrame
        The training set should be a training set in cross-validation.
    s_count: int, default 200
        The sample count for one theme if the sample_count is lower than s_count, then oversampling to s_count.
    
    Returns
    -------
    DataFrame
        Return balanced training set
    """    
    # balance
    sample_count = df_train.theme.value_counts().min() # get the minority theme

    if sample_count > s_count:
        s_count = sample_count
        print(f'Balancing: Get the random {s_count} records in each Theme...')

    ds_train_balanced = df_train.groupby('theme').apply(
        lambda df: df.sample(s_count, random_state=42, replace=True) #df.sort_values('count', ascending=False)[:sample_count]
        ).reset_index(drop=True)

    return ds_train_balanced

"""
def get_balanced_training_set(df_train, s_count=200):
    "
    Prepare data source for theme classification parallelly.
    
    Parameters
    ----------
    df_train : DataFrame
        The training set should be a training set in cross-validation.
    s_count: int, default 200
        The sample count for one theme if the sample_count is lower than s_count, then oversampling to s_count.
    
    Returns
    -------
    DataFrame
        Return balanced training set
    "    
    # balance: get the records with more texts
    sample_count = df_train.theme.value_counts().min() # get the minority theme

    # downsampling
    if sample_count > s_count:
        print(f'Balancing: Get the top {sample_count} records in each Theme...')

        ds_train_balanced = df_train.groupby('theme').apply(
                lambda df: df.sort_values('count', ascending=False)[:sample_count]
            ).reset_index(drop=True)
        
        return ds_train_balanced
    
    # oversampling
    print(f'{sample_count} records in each Theme is too small, we do oversampling to minorities...')

    sample_count = s_count

    samples = []

    for theme, group in df_train.groupby('theme'):
        if group.shape[0] > sample_count:
            # get top 100 by word count
            samples.append(group.sort_values('count', ascending=False)[:sample_count])
        else:
            # oversampling
            samples.append(group.sample(sample_count, random_state=42, replace=True))

    ds_train_balanced = pd.concat(samples, ignore_index=True)

    return ds_train_balanced
"""


# In[52]:


def evaluate_classification(ds, t=0, min_df=4, max_df=1000, alpha=0.05, s_count=200):
    """
    Evaluate a estimater with the given hypterparameters by 5 fold cross validation.
    Show the classification evaluate resultsand prediction probablity threshold. 
    The estimator is MultinomialNB.
    
    Parameters
    ----------
    ds: DataFrame
        The data source contain columns: content and theme(the df_train from func prepare_data_source()).
    t : float, default 0
        Threshold for prediction probablity. If the maximum prob is lower then t, the class will be marked as -1.
    min_df : int or float, default 4
    max_df : int or float, default 1000
    alpha : float, default 0.05
        The hyperparameter of MultinomialNB.
    s_count : int, default 200
        The minimum sample size requrement.
    
    Returns
    -------
    acc : float
    p : float
    r : float
    f1 : float
    """
    # encode target
    print('encode target...')
    le = LabelEncoder()

    target = le.fit_transform(ds['theme'])

    # 5 folds cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)

    results = pd.DataFrame({'actual': target, 'predicted': 0})
    
    print('cross validation...')
    
    for train_indices, test_indices in skf.split(ds['content'], target):
        # test
        test_data = ds.loc[test_indices, 'content']
        y_test = target[test_indices]
        
        # balance training set
        balanced_train_set = get_balanced_training_set(ds.iloc[train_indices], s_count)
        
        train_data = balanced_train_set['content'] #ds.loc[train_indices, 'content']
        y_train = le.transform(balanced_train_set['theme']) #target[train_indices]
          
        # generate features
        print('generate features...')

        tfidf = tfidf_vectorizer.fit(train_data)

        X_train = pd.DataFrame(tfidf.transform(train_data).toarray(), columns=tfidf.get_feature_names_out())
        
        print(f'features: {X_train.shape}')

        # estimator fit
        print('fit estimator...')
        clf = MultinomialNB(alpha=alpha)
        #clf = SVC(decision_function_shape='ovo')
        
        clf.fit(X_train, y_train)

        # predict
        print('predict...')
        # features
        X_test = pd.DataFrame(tfidf_vectorizer.transform(test_data).toarray(), columns=tfidf.get_feature_names_out())
        
        y_predicted = None
        
        if t == 0:
            y_predicted = clf.predict(X_test)
        else:
            y_prob = clf.predict_proba(X_test)

            # handle threshold
            max_prob = y_prob.max(axis=1)
            y_predicted = y_prob.argmax(axis=1)

            y_predicted[max_prob < t] = -1

        results.loc[test_indices, 'predicted'] = y_predicted

    # evaluation
    print('the records failed to predict...')
    display(ds[results.predicted==-1].head())
    
    results = results[results.predicted>-1]
    print(f'prediction percentage: {results.shape[0]/ds.shape[0]}')
    
    y_actual = results.actual.values
    y_predicted = results.predicted.values

    print('Confusion Matrix:')
    cm = confusion_matrix(y_actual, y_predicted, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
    print(pd.Series(le.inverse_transform(clf.classes_)))

    acc = accuracy_score(y_actual, y_predicted)
    p = precision_score(y_actual, y_predicted, average='macro')
    r = recall_score(y_actual, y_predicted, average='macro')
    f1 = f1_score(y_actual, y_predicted, average='macro')

    print(f'prediction accuracy is {acc}; precision:{p}; recall: {r}; f1 score： {f1}')
    
    return acc, p, r, f1



# In[5]:


def create_model(train_set, t, min_df, max_df, alpha, s_count):
    """
    Fit the estimator on train_set with the tuned hyperparameters.
    
    Parameters
    ----------
    train_set : DataFrame
    t : float
    min_df : int
    max_df : int
    alpha : float
    s_count : int
    
    Returns
    -------
    model : dict
        Items are lable encoder, tfidf vectorizer, and classifier.
    """          
    print('generate features...')
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    
    # balance training set
    balanced_train_set = get_balanced_training_set(train_set, s_count)
    
    # features
    tfidf = tfidf_vectorizer.fit(balanced_train_set['content'])

    X_train = pd.DataFrame(tfidf.transform(balanced_train_set['content']).toarray(), columns=tfidf.get_feature_names_out())
    print(f'features: {X_train.shape}')
    
    print('encode target...')
    le = LabelEncoder()

    target = le.fit_transform(balanced_train_set['theme'])

    # estimator
    print('fit estimator...')
    clf = MultinomialNB(alpha=alpha)
    
    clf.fit(X_train, target)
    
    # save model
    model = {
        'le': le,
        'clf': clf,
        'vectorizer': tfidf_vectorizer,
        'tfidf': tfidf
    }
    
    return model
    


# In[6]:


def save_model(train_set, save_name, t, min_df, max_df, alpha, s_count=200):
    """
    Save the estimator with the best hyperparameter.
    
    Parameters
    ----------
    train_set : DataFrame
    save_name : str
    t : float
    min_df : int
    max_df : int
    alpha : float
    s_count : int, default 200
    """
       
    # directory
    dir_path = os.path.join(data_path, 'models')
    save_path = os.path.join(dir_path, save_name)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    model = create_model(train_set, t, min_df, max_df, alpha, s_count)
    
    pickle.dump(model, open(save_path, 'wb'))
    


# In[10]:


def predict(df_unseen, model, t=0):
    """
    Predict the themes of df_unseen.
    
    Parameters
    ----------
    df_unseen : DataFrame
    model : str or dict
        If it is str, it is model path, need to load the model from the path.
    t : float
        The threshold parameter
    
    Returns
    -------
    df_unseen : DataFrame
        Update theme column of df_unseen.
    """    
    df_unseen = df_unseen.copy() 
    
    # get model
    if type(model) == str:
        model = pickle.load(open(os.path.join(data_path, 'models', model), 'rb'))
    
    tfidf = model['tfidf']
    le = model['le']
    clf = model['clf']
    tfidf_vectorizer = model['vectorizer']

    # predict
    print('predict...') # transfer to unicode cuz there is nan
    X_unseen = pd.DataFrame(tfidf_vectorizer.transform(df_unseen['content'].values.astype('U')).toarray(), columns=tfidf.get_feature_names_out())
    
    
    # deal with '<Unknown>'    
    if t == 0:
        y_predicted = clf.predict(X_unseen)
        
        df_unseen['theme'] = le.inverse_transform(y_predicted)
        
    else:
        print('deal with unknown target...')
        y_prob = clf.predict_proba(X_unseen)

        # handle threshold
        max_prob = y_prob.max(axis=1)
        y_predicted = y_prob.argmax(axis=1)

        y_predicted[max_prob < t] = -1
        
        le_dict = dict(zip(le.transform(le.classes_), le.classes_))
        le_dict[-1] = '<Unknown>'
        
        df_unseen['theme'] = y_predicted
        
        df_unseen['theme'] = df_unseen.theme.map(le_dict)
    
    return df_unseen


# In[113]:


def generalization_performance(train_set, test_set, t, min_df, max_df, alpha, s_count):
    """
    Show generalization performance.
    
    Parameters
    ----------
    train_set : DataFrame
    df_unseen : DataFrame
    t : float
    min_df : int
    max_df : int
    alpha : float
    s_count : int
    """
    # get predicted test
    model = create_model(train_set, t, min_df, max_df, alpha, s_count)
    predicted_test_set = predict(test_set, model, t)
    
    y_actual = test_set['theme']
    y_predicted = predicted_test_set['theme']
    
    # evaluation
    count_unpredicted = (y_predicted=='<Unknown>').sum()
    
    print(f'{count_unpredicted} records failed to predict...')
    
    print(f'prediction coverage: {1 - count_unpredicted / y_predicted.shape[0]}')
    
    le = LabelEncoder()
    le.fit(y_actual.append(y_predicted))
    
    y_actual = le.transform(y_actual)
    y_predicted = le.transform(y_predicted)

    acc = accuracy_score(y_actual, y_predicted)
    p = precision_score(y_actual, y_predicted, average='macro')
    r = recall_score(y_actual, y_predicted, average='macro')
    f1 = f1_score(y_actual, y_predicted, average='macro')
    
    print(acc, p, r, f1)
    


# # Classification: Tuning
# 
# 

# In[50]:


def get_accuracy(results, para_options):
    """
    Get accuracy Series from results list.
    
    Parameters
    ----------
    results : list of tuples
        The item is the return value of function evaluate_classification()
    para_options : list of int
        The item is candidates of the hyperparameters to be tuned.
    
    Returns
    -------
    df : DataFrame
        Columns are metrics, rows are parameter available.
    best : int or float
        The best value in the candidates.
    """
    if len(para_options) == 0:
        return None, None
    
    df = pd.DataFrame(results, index=para_options, columns=['accuracy', 'precision', 'recall', 'f1-score'])
    
    best = para_options[df.accuracy.argmax()]

    print(f'The best parameter is {best}.')
    
    return df, best


def tune_hyperparameters(df_train, t_l, min_df_l, max_df_l, alpha_l, s_count_l):
    """
    Get the best min_df and max_df for Vectorizer and parameters for estimator.
    The parameter will be tuned separately, will not be tuned in a parameter combination way.
    
    Parameters
    ----------
    df_train : DataFrame
    t_l : list of float
    min_df_l : list of int
    max_df_l : list of int
    alpha_l : list of float
    s_count_l : list of int
    
    Returns
    -------
    tuple of Series
        The item is accuracy Series for each parameter.
    tuple of Series
        The item is precision Series for each parameter.
    tuple of DataFrame
        Each item is the performance result.
    tuple of float
        Each item is the best value of the parameter.
    """
    results = []
    
    # tune min_df
    print('tune min_df...')
    for min_df in min_df_l:
        print(f'min_df={min_df}')
        
        results.append(evaluate_classification(df_train, 0, min_df, 50)) # in case memory issue for permit
        
    min_df_metrics, best_min_df = get_accuracy(results, min_df_l)
    
    # tune max_df
    results = []
    
    print('tune max_df...')
    
    for max_df in max_df_l:
        print(f'max_df={max_df}')
        results.append(evaluate_classification(df_train, 0, 10, max_df))
        
    max_df_metrics, best_max_df = get_accuracy(results, max_df_l)
    
    # tune alpha
    results = []
    
    print('tune alpha...')
    
    for alpha in alpha_l:
        print(f'alpha={alpha}')
        results.append(evaluate_classification(df_train, 0, 5, 50, alpha))
        
    alpha_metrics, best_alpha = get_accuracy(results, alpha_l)
    
    # tune thereshold for 'unknown'
    results = []
    
    print('tune threshold...')
    
    for t in t_l:
        print(f't={t}')
        results.append(evaluate_classification(df_train, t))
        
    t_metrics, best_t = get_accuracy(results, t_l)
    
    # tune sample count
    results = []
    
    print('tune sample count...')
    
    for s_count in s_count_l:
        print(f's_count={s_count}')
        results.append(evaluate_classification(df_train, 0, 5, 50, 1, s_count))
        
    s_metrics, best_s = get_accuracy(results, s_count_l)
    
    return (min_df_metrics, max_df_metrics, alpha_metrics, t_metrics, s_metrics),\
           (best_min_df, best_max_df, best_alpha, best_t, best_s)


# In[188]:


def save_event_with_theme(df_event_original, train, unseen, filepath):
    """
    Add the predicted theme column to the original event table and save to local.
    
    Parameters
    ----------
    df_event_original : DataFrame
    train : DataFrame
    unseen : DataFrame
        The column theme has beed predicted.
    filepath : str
    
    Returns
    -------
    DataFrame
    """
    # combine theme
    df_theme = pd.concat([train[['index', 'theme']], unseen[['index', 'theme']]]).set_index('index')

    # merge into original event
    if 'theme' in df_event_original.columns:
        df_event_original = df_event_original.drop('theme', axis=1)
        
    df_event_original = df_event_original.join(df_theme)

    # save to local
    df_event_original.to_excel(os.path.join(sub_event_path, filepath), index=False, freeze_panes=(1, 2))
    
    return df_event_original
    


# In[28]:


def show_wordcloud_for_event(df, prefix=''):
    """
    Generate and show wordcloud for each theme of the given event.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame contains at least columns: theme and content.
    prefix : str, default ''
        The prefix for wordcloud pictures.
    """
    print('Prevalent words in each Theme')

    # all data
    df_freq, freq_dict = get_frequency_table(df['content'])
    show_wordcloud(freq_dict, title = prefix + 'All data')

    # by theme
    for t in df.theme.unique():
        df_freq, freq_dict = get_frequency_table(df.loc[df.theme == t, 'content'])

        show_wordcloud(freq_dict, title = prefix + t)
        


# In[ ]:





# In[ ]:




