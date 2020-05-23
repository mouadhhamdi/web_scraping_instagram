import pandas as pd
import numpy as np
import os
import logging
import re
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import emoji
import plotly.express as px
from collections import defaultdict
from nltk.corpus import stopwords
import string
import warnings
warnings.filterwarnings("ignore")
import gzip
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import cufflinks as cf
from tqdm import tqdm_notebook as tqdm
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_rows', None)
PATH_PUBLICATIONS_OMEGA = './omega'
PATH_PUBLICATIONS_CARTIER = './cartier'
PATH_PUBLICATION_ROLEX = './rolex'

PATH_COMMENTS_OMEGA = './link_insta/omega-comments.txt'
PATH_COMMENTS_CARTIER = './link_insta/cartier-comments.txt'
PATH_COMMENTS_ROLEX = './link_insta/rolex-comments.txt'

PATH_ALL_COMMENTS = 'all_comments.csv'

emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    
# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')


# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

def get_image_code(insta_link):
    try:
        insta_image_code = insta_link.split('/')[-2]
    except:
        logger.error("get_image_code" + insta_image_code)
        
    return insta_image_code

def create_instaloader_file(insta_image_code):
    try:
        logger.info("building files for" + insta_image_code)
        os.system(" instaloader --comments -- -" + insta_image_code)
    except:
        logger.error("create_instaloader" + insta_image_code)


def web_scrap_comments(insta_link):
    insta_image_code = get_image_code(insta_link)
    create_instaloader_file(insta_image_code)



def process_comments(comment, with_emoji=False):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(comment)
    
    # convert text to lower-case
    comment = comment.lower() 
    # remove URLs
    comment = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', comment) 
    # remove usernames
    comment = re.sub('@[^\s]+', '', comment) 
    # remove the # in #hashtag
    comment = re.sub(r'#([^\s]+)', r'\1', comment) 
    if not with_emoji:
        comment = emoji_pattern.sub(r'', comment)
    else:
        # transform emoji to str between ::
        comment = emoji.demojize(comment)
          
    # split emoji names 
    comment = ' '.join(re.split('[- _ :]', comment))
    comment = comment.replace('’', ' ')
    # tokenize the words in comment
    word_tokens = word_tokenize(comment)    
    
    # construct filtered list of words
    #filtered_comment = [w for w in word_tokens if not w in stop_words]
    output_comment = []
    for w in word_tokens:
    #check tokens against punctuations
        if w not in string.punctuation :#and w not in stop_words:
            output_comment.append(w)
            
    return ' '.join(output_comment)


def generate_box_plot(comments_df, company):
    return comments_df[comments_df['Company name']==company][['pos_sia_without_emoji', 'neg_sia_without_emoji',
                                                           'neu_sia_without_emoji']].\
    rename(columns={'pos_sia_without_emoji':'positive', 'neg_sia_without_emoji':'negative', 
                   'neu_sia_without_emoji':'neutral'}).\
                        iplot(kind='box', asFigure=True, theme='white',gridcolor='white',
                                       bargap=0.5, xTitle='box plot', 
                                        title=company +' comments boxplot')


def get_distributions(comment_df, company):
    return comment_df[comment_df['Company name']==company][['pos_sia_without_emoji', 'neg_sia_without_emoji',
                                                           'neu_sia_without_emoji', 'agg_sia_without_emoji']].\
    rename(columns={'pos_sia_without_emoji':'positive', 'neg_sia_without_emoji':'negative', 
                       'neu_sia_without_emoji':'neutral', 'agg_sia_without_emoji': 'aggregation'}).\
    iplot(asFigure=True, kind="histogram", theme='white',gridcolor='white',
                                bargap=0.01, xTitle='distribution', yTitle='count', 
                                title=company+' sentiment distribution', subplots=True, bins=10)

def get_comment_length_distribution(comment_df, company):
    return comment_df[comment_df['Company name']==company].\
                    rename(columns={'comment_cleaned_length_without_emoji':'length'})[['length']].\
                        iplot(kind='hist', bins=100, asFigure=True, theme='white',gridcolor='white',
                                        bargap=0.01, xTitle='length', yTitle='count', 
                                        title=company+' distribution of comments length')
                                
def get_activity_per_day(comment_df, company):
    comment_df = comment_df[comment_df['Company name']==company]
    return comment_df.set_index('comment date').resample('d').sum()[['pos_sia_emoji']].rename(columns={'pos_sia_emoji':'reactions'}).iplot(
    asFigure=True, theme='white',gridcolor='white',
                            bargap=0.01, xTitle='date', yTitle='count', 
                            title=company+' comments timeline', mode='lines',
        text='reactions', bestfit=True, bestfit_colors=['blue'], world_readable=True, fill=True)


def important_words_per_company(all_comments_df, company):
    all_comments_df = all_comments_df[all_comments_df['Company name']==company]
    corpus=[]
    corpus=[word for w in all_comments_df.comment_cleaned_without_emoji.str.split() for word in w]
    # count of word without stop words
    word_dic = defaultdict(int)
    for word in corpus:
        if word not in set(stopwords.words('english')) and word != '...':
            word_dic[word] += 1
    
    # get the world count
    word_count = sorted(word_dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    word, count = zip(*word_count)
    word_count_df = pd.DataFrame(word_count, columns=['word', 'count'])
    #fig = px.bar(word_count_df, x='word', y='count')
    return word_count_df[:10].iplot(kind='bar', asFigure=True, theme='white',gridcolor='white',
                                       bargap=0.5, yTitle='count', xTitle='words', 
                                        title=company+' most common words', x='word')



def prepare_comments_df(all_comments_df):
    # apply cleaning on the comments
    all_comments_df['comment_cleaned_with_emoji'] = all_comments_df.comment.apply(lambda row: 
                                                                                process_comments(row, True))
    # apply cleaning on the comments
    all_comments_df['comment_cleaned_without_emoji'] = all_comments_df.comment.apply(lambda row: 
                                                                                process_comments(row, False))
    # get the length of the cleaned comment
    all_comments_df['comment_cleaned_length_with_emoji'] = all_comments_df.comment_cleaned_with_emoji.apply(lambda row: 
                                                                                    len(row.split()))
    all_comments_df['comment_cleaned_length_without_emoji'] = all_comments_df.comment_cleaned_without_emoji.apply(lambda row:
                                                                                        len(row.split()))
    sia = SIA()
    all_comments_df['pos_sia_emoji'] = all_comments_df['comment_cleaned_with_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['pos'])
    all_comments_df['neg_sia_emoji'] = all_comments_df['comment_cleaned_with_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['neg'])
    all_comments_df['neu_sia_emoji'] = all_comments_df['comment_cleaned_with_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['neu'])
    all_comments_df['agg_sia_emoji'] = all_comments_df['comment_cleaned_with_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['compound'])
    all_comments_df['pos_sia_without_emoji'] = all_comments_df['comment_cleaned_without_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['pos'])
    all_comments_df['neg_sia_without_emoji'] = all_comments_df['comment_cleaned_without_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['neg'])
    all_comments_df['neu_sia_without_emoji'] = all_comments_df['comment_cleaned_without_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['neu'])
    all_comments_df['agg_sia_without_emoji'] = all_comments_df['comment_cleaned_without_emoji'].apply(lambda 
                                                        comment:sia.polarity_scores(comment)['compound'])
    all_comments_df['classification_with_emoji'] = all_comments_df.apply(lambda row: 'pos' 
                                                                        if row['agg_sia_emoji']>=0 else 'neg', axis=1)
    all_comments_df['classification_without_emoji'] = all_comments_df.apply(lambda row: 'pos' 
                                                                if row['agg_sia_without_emoji']>=0 else 'neg', axis=1)
    all_comments_df['classification_with_emoji'].iplot(kind='hist', asFigure=True, theme='white',gridcolor='white',
                                        bargap=0.5, xTitle='classification', yTitle='count', 
                                            title='Omega comments classification with emoji')
    all_comments_df['classification_without_emoji'].iplot(kind='hist', asFigure=True, theme='white',gridcolor='white',
                                        bargap=0.5, xTitle='classification', yTitle='count', 
                                            title='Omega comments classification without emoji')
    # remove empty commenths
    all_comments_df = all_comments_df[all_comments_df.comment_cleaned_length_without_emoji>0]
    all_comments_df['comment date'] = pd.to_datetime(all_comments_df['comment date'])
    # remove hours, minutes, seconds
    all_comments_df["comment date day"] = all_comments_df["comment date"].dt.strftime("%m-%d-%y")
    all_comments_df["comment date hour"] = all_comments_df["comment date"].dt.strftime("%H:%M:%S")
    return all_comments_df



def process_comments_stop_words(comment):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(comment)
    
    # tokenize the words in comment
    word_tokens = word_tokenize(comment)    
    
    # construct filtered list of words
    filtered_comment = [w for w in word_tokens if not w in stop_words]
    output_comment = []
    for w in word_tokens:
    #check tokens against punctuations
        if w not in string.punctuation and w not in stop_words:
            output_comment.append(w)
            
    return ' '.join(output_comment)


def important_ngram_per_company(all_comments_df, company):
    all_comments_df = all_comments_df[all_comments_df['Company name']==company]
    
    all_comments_df['comment_cleaned_without_emoji'] = all_comments_df['comment_cleaned_without_emoji']\
            .apply(lambda r: process_comments_stop_words(r))
    
    
    word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(all_comments_df['comment_cleaned_without_emoji'])
    frequencies = sum(sparse_matrix).toarray()[0]
    ngram_df = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), 
                 columns=['frequency']).sort_values('frequency', ascending=False).reset_index().rename(
                    columns={'index':'ngram'})
    #fig = px.bar(ngram_df[:10], x='ngram', y='frequency')
    return ngram_df[:10].iplot(kind='bar', asFigure=True, theme='white',gridcolor='white',
                                       bargap=0.5, yTitle='count', xTitle='words', 
                                        title=company+' most common 2-grams', x='ngram')

def get_word_dict(all_comments_df, company):
    all_comments_df = all_comments_df[all_comments_df['Company name']==company]
    corpus=[]
    corpus=[word for w in all_comments_df.comment_cleaned_without_emoji.str.split() for word in w]
    # count of word without stop words
    word_dic = defaultdict(int)
    for word in corpus:
        if word not in set(stopwords.words('english')) and word != '...':
            word_dic[word] += 1
            #print(word)
            if word != porter_stemmer.stem(word):
                word_dic[porter_stemmer.stem(word)] += 1
                #print(porter_stemmer.stem(word))
            if word != wordnet_lemmatizer.lemmatize(word) and wordnet_lemmatizer.lemmatize(word) != porter_stemmer.stem(word):
                word_dic[wordnet_lemmatizer.lemmatize(word)] += 1
                #print(wordnet_lemmatizer.lemmatize(word))
    
    return word_dic

def word_freq(dict_words_omega, dict_words_rolex, dict_words_cartier, word):
    if word in dict_words_omega.keys():
        count_omega = dict_words_omega[word]
    else:
        count_omega = 0
    if word in dict_words_rolex:
        count_rolex = dict_words_rolex[word]
    else:
        count_rolex = 0
    if word in dict_words_cartier:
        count_cartier = dict_words_cartier[word]
    else:
        count_cartier = 0
    #print("Omega {word} count: {count_omega} \n Rolex {word} count: {count_rolex} \n Cartier {word} count: {count_cartier}".format(word=word, 
    #count_omega=count_omega, count_rolex=count_rolex, count_cartier=count_cartier))
    return pd.DataFrame(data=[[word, count_omega, count_rolex, count_cartier]], columns=['word', 'omega count', 'rolex count', 'cartier count'])



