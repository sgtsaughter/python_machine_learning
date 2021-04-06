# load model and ask to make predictions
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions