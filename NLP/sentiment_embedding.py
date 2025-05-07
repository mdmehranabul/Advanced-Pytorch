#%%
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sentence_transformers import SentenceTransformer
# %%
twitter_file='data/Tweets.csv'
df=pd.read_csv(twitter_file).dropna()
df
# %%
cat_id={'neutral':1,
        'negative':0,
        'positive':2}
df['class']=df['sentiment'].map(cat_id)
# %%
