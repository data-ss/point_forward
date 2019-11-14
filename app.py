import flask
import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd

app = flask.Flask(__name__, template_folder="")

#-------- MODEL GOES HERE -----------#

# pipe = pickle.load(open("model/pipe.pkl", "rb"))

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.sparse import coo_matrix
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
# import pickle
from sklearn.pipeline import make_pipeline

# training data
df_indeed_list = pd.read_csv("data/indeed_data_scientist_list.csv")
df_linkedin_list = pd.read_csv("data/linkedin_data_scientist_list.csv")
df_list = pd.concat([df_indeed_list, df_linkedin_list], axis=0)
df_list.drop_duplicates(subset=["title", "company", "text"], keep="first", inplace=True)
df_list.dropna(subset=["text"], inplace=True)
df_list.reset_index(inplace=True, drop=True)

##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "show", "result", "large", "also", "iv", "one", "owner", "two", "new", "previously", "cambridge", "shown", "work", "technology", "de", "business", "product", "et", "asset", "required", "engineer", "opportunity", "within", "including", "tool", "create", "etc"]
stop_words = stop_words.union(new_words)

# cleaning the text data
# pass corpora as a list
def cleaner(corpora):
    corpus = []
    for i in range(len(corpora)):
        #Remove punctuations
        text = re.sub(r'[^a-zA-Z]', ' ', corpora[i])

        #Convert to lowercase
        text = text.lower()

        #remove tags
        text=re.sub(r"&lt;/?.*?&gt;"," &lt;&gt; ",text)

        # remove special characters and digits
        text=re.sub(r"(\\d|\\W)+"," ",text)

        ##Convert to list from string
        text = text.split()

        ##Stemming
        ps=PorterStemmer()    #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        corpus.append(text)
    return corpus

corpus = cleaner(corpora)

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)

#Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)

# get feature names
feature_names=cv.get_feature_names()


test = ["""
    Work closely with business owners to identify opportunities and serve as an ambassador for data science Design and deliver enterprise analytic solutions to our customers Develop powerful business insights from social, marketing and industrial data using advanced machine learning techniques
        Build complex statistical models that learn from and scale to petabytes of data Work in a highly interactive, team-oriented environment with Big Data developers and Visualisation experts  Graduate level qualification in a relevant technical field (Computer Science, Engineering, Applied Mathematics/Statistics, Operations Research) ideally with a specialization in data mining/machine learning
Up to 5 years' experience of working in analytical environments
Deep expertise in analytical tools such as R/Matlab/SAS/Stata
Experience of scripting languages such as Python/Ruby/PHP etc.
Experience of relational databases and usage of SQL
Experience of Object Orientated programming via Java/C++
Experience of Big data technologies (Hadoop, HIVE, PIG, Spark etc.) and unstructured data
Demonstrated track record of manipulation of large volume, high frequency data for analytical purposes on a Big Data platform
Demonstrated experience of developing and implementing statistical models (predictive & descriptive)
Demonstrated experience in delivering high quality, high impact analytical solutions to business problems
"""]

doc = cleaner(test)
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

### RESULTS ###
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
###

# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])


#-------- ROUTES GO HERE -----------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = request.form






    predicted = pipe.predict(new)[0]
    if predicted == 1:
        prediction = "YOUR LEFT STROKE JUST WENT VIRAL!!"
    else:
        prediction = "Sit down, be humble. Probably not gonna go viral."
    # prediction = '{:,.2f}%'.format(prediction)
    return render_template('index.html', prediction=prediction)




if __name__ == '__main__':
    '''Connects to the server'''

    app.run(port=5000, debug=True)
