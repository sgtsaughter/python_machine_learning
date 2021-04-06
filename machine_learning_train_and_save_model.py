# Build, train, and save model to a file. Then load model and ask to make predictions
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv('music.csv')
# By convention, we use X to represent Input data, and y to represent output data.
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model.fit(X, y)

# Create model in a new file called music-recommender.joblib
joblib.dump(model, 'music-recommender.joblib')

