# This is basically the heart of my flask 
from flask import Flask, render_template, request
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
# import xgboost


app = Flask(__name__)
#change the paths to relative
with open('Model/tifidfmodel.pkl','rb') as tfidf:
    tfidf_model = pickle.load(tfidf)
with open('Model/logrmodel.pkl','rb') as logr:
    logr_model = pickle.load(logr)
with open('Model/item_final_rating.pkl','rb') as item_rating:
    item_final_rating = pickle.load(item_rating)

#function to get reviews based on product name
#sample =needs data load
def getProductReview(name,sample):
  out=0
  count=0
  final=0
  df_value=sample[sample['name'] ==name]
  df_value=df_value['reviews_text']
  for i in df_value:
    out=out+getSentimentScore(tfidf_model.transform([i]))
    count=count+1
  final=out/count
  return final

def getSentimentScore(review_text):
  intScore=0
  score=logr_model.predict(review_text)
  if score=='Positive':
    intScore=1
  if(score=='Negetive'):
    intScore=-1
  return intScore

@app.route('/')
def home():
    return render_template('index.html')
#needs item_final_rating matrix and W
@app.route('/predict',methods=['POST'])
def predict():
    User = request.form.get('User')
#    Input = [[User]]
    print("user input*******",User)
    sample = pd.read_csv("Data/sample30.csv")
    d = item_final_rating.loc[User].sort_values(ascending=False)[0:20]
    user_choices=pd.DataFrame(data=d,columns=[User])
    user_choices = user_choices.reset_index()
    user_choices['sentiment_score']=user_choices['name'].apply(lambda x:getProductReview(x,sample))
    final_recommend=user_choices.sort_values('sentiment_score',ascending=False)
    final_recommend=final_recommend.head(5)
    prediction = final_recommend
    return render_template('index.html', OUTPUT=str(prediction.name))

if __name__ == "__main__":
    app.run(debug=True)