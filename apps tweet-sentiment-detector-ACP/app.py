from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app=Flask(__name__)
@app.route('/predict',methods=['POST'])



def predict():
    df = pd.read_excel('TweetsFinal.xlsx')
    df['label'] = df['sentiment'].map({'positive': 0, 'negative': 1})
    X = df["Feed"]
    y = df["label"]
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)
    if request.method=='POST':
        v2=request.form['v2']
        data=[v2]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction)
@app.route('/')
def home():
	return render_template('home.html')

if __name__=='__main__':
   app.run(debug=False)
