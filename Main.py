import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords # the for of in with
from nltk.stem.porter import PorterStemmer # loved loving == love
from sklearn.feature_extraction.text import TfidfVectorizer # loved = [0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
news_df = pd.read_csv('train.csv')
news_df.head()

news_df.shape

news_df.isna().sum()

news_df = news_df.fillna(' ')

news_df.isna().sum()

news_df['content'] = news_df['author']+" "+news_df['title']
news_df


news_df['content']

ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
news_df['content'] = news_df['content'].apply(stemming)
news_df['content']

X = news_df['content'].values
y = news_df['label'].values
print(X)
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)
X_train.shape

X_test.shape

model = LogisticRegression()
model.fit(X_train,y_train)
LogisticRegression()
train_y_pred = model.predict(X_train)
print("train accurracy :",accuracy_score(train_y_pred,y_train))

test_y_pred = model.predict(X_test)
print("train accurracy :",accuracy_score(test_y_pred,y_test))



input_data = X_test[20]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print('Fake news')
else:
    print('Real news')

news_df['content'][20]
