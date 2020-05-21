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

# Imports
import gzip
import os
import pandas as pd
import numpy as np
# Standard plotly imports

from tqdm import tqdm_notebook as tqdm


pd.set_option('display.max_rows', None)
PATH_PUBLICATIONS_OMEGA = './omega'
PATH_PUBLICATIONS_CARTIER = './cartier'
PATH_PUBLICATION_ROLEX = './rolex'

PATH_COMMENTS_OMEGA = './link_insta/omega-comments.txt'
PATH_COMMENTS_CARTIER = './link_insta/cartier-comments.txt'
PATH_COMMENTS_ROLEX = './link_insta/rolex-comments.txt'

PATH_ALL_COMMENTS = 'all_comments.csv'
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
        logger.info("building files for", insta_image_code)
        os.system(" instaloader --comments -- -" + insta_image_code)
    except:
        logger.error("create_instaloader" + insta_image_code)


def web_scrap_comments(insta_link):
    insta_image_code = get_image_code(insta_link)
    create_instaloader_file(insta_image_code)

