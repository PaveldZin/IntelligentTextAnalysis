import re
import nltk
import pandas as pd
from string import punctuation
from pymystem3 import Mystem
from nltk.corpus import stopwords
from joblib import Parallel, delayed

nltk.download('stopwords')

to_remove = stopwords.words('russian') + ['RT', '', ' ', '\n']

def clean_text(text):
    text = re.sub(r'@\w+', '', text) #remove mentions
    text = re.sub(r'#\w+', '', text) #remove hashtags
    text = re.sub(r'http\S+', '', text) #remove urls
    text = re.sub('[{}]'.format(punctuation), ' ', text) #remove punctuation
    text = re.sub(r'\d+', '', text) #remove digits
    text = re.sub(r'\s+', ' ', text) #remove extra whitespaces
    return text.strip()

def lemmatize(text):
    mystem_analyzer = Mystem(grammar_info=False)
    lemmas = mystem_analyzer.lemmatize(text)
    lemmas[-1] = lemmas[-1].rstrip()
    return [word for word in lemmas if word not in to_remove]

def preprocess(text):
    text = clean_text(text)
    text = lemmatize(text)
    return text

def process_batch(text):
    merged_text = " sep ".join(text)

    doc = []
    res = []

    for t in preprocess(merged_text):
        if t.strip() != 'sep':
            doc.append(t)
        else:
            res.append(doc)
            doc = []
    res.append(doc)
    return res

def parallel_preprocess(data, batch_size=1000):
    texts = data.values
    text_batch = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]
    processed_texts = Parallel(n_jobs=-1, backend="threading")(delayed(process_batch)(t) for t in text_batch)
    combined_texts = [' '.join(text) for batch in processed_texts for text in batch]
    return pd.Series(combined_texts, index=data.index)
