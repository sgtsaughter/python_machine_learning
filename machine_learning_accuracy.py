# Displaying accuracy of a suggested music genre model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
# By convention, we use X to represent Input data, and y to represent output data.
X = music_data.drop(columns=['genre'])
y = music_data['genre']
# Allocate 20% of our data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score