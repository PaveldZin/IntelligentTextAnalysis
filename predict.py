import joblib
import pandas as pd
from preprocess import parallel_preprocess
vectorizer = joblib.load('pickle/vectorizer.pkl')
model = joblib.load('pickle/SVM.pkl')

#data is a pandas series containing texts to predict labels for
def predict(data):
    processed_data = parallel_preprocess(data)
    vec_data = vectorizer.transform(processed_data)
    return model.predict(vec_data)