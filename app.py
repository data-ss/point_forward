import flask
# import pickle
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.sparse import coo_matrix
import re
import nltk
from wordcloud import WordCloud
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from flask_table import Table, Col
from gazpacho import get, Soup
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
import random
# import sqlite3
# from sqlite3 import Error

options = Options()
options.headless = True
browser = Firefox(options=options)


app = flask.Flask(__name__, template_folder="")

#-------- MODEL GOES HERE -----------#

# training data
df_indeed_list = pd.read_csv("model/data/indeed_data_scientist_list.csv")
df_linkedin_list = pd.read_csv("model/data/linkedin_data_scientist_list.csv")
df_ux = pd.read_csv("model/data/indeed_ux_ui_designer_list.csv")
df_list = pd.concat([df_indeed_list, df_linkedin_list, df_ux], axis=0)
df_list.drop_duplicates(subset=["title", "company", "text"], keep="first", inplace=True)
df_list.dropna(subset=["text"], inplace=True)
df_list.reset_index(inplace=True, drop=True)

##Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["using", "company", "monitor", "outcome", "show", "result", "large", "also", "iv", "one", "owner", "two", "new", "previously", "cambridge", "shown", "work", "technology", "de", "business", "product", "et", "asset", "required", "engineer", "opportunity", "within", "including", "tool", "create", "etc"]
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

corpus = cleaner(df_list["text"])

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

### SCRAPER
def indeed_scraper(job_title, jobs_per_page, search_radius, pages):
    '''
    1. Automatically uses "+" as a space delimiter when entering job_title

    '''
    job_title = job_title.replace(" ", "+")
    url = f"https://ca.indeed.com/jobs?q={job_title}&l=Toronto%2C+ON&limit={jobs_per_page}&radius={search_radius}&start={pages}"
    html = get(url)
    soup = Soup(html)
    jerbs_index = soup.find("td", {"id": "resultsCol"})

    jerbs_indeed = []
    for i in range(len(jerbs_index.find("div", {"class": "title"}, mode="all"))):
        list_1 = {}
        list_1["title"] = jerbs_index.find("div", {"class": "title"}, mode="all")[i].find("a").attrs["title"]
        if jerbs_index.find("span", {"class": "company"}, mode="all")[i].text == "":
            list_1["company"] = jerbs_index.find("div", {"class": "sjcl"}, mode="all")[i].find("a")[0].text
        else:
            list_1["company"] = jerbs_index.find("span", {"class": "company"}, mode="all")[i].text
        time.sleep(random.randint(1, 10) / 10)
        list_1["url"] = "https://ca.indeed.com"+jerbs_index.find("div", {"class": "title"}, mode="all")[i].find("a").attrs["href"]
        html_jerb = get(list_1["url"])
        soup_jerb = Soup(html_jerb)
        list_1["text"] = BeautifulSoup(" ".join([i.html for i in soup_jerb.find("div", {"id": "jobDescriptionText"}, mode="all")[0].find("li", mode="all")])).get_text(" ").replace("\n", " ")
        jerbs_indeed.append(list_1)
        time.sleep(random.randint(1, 10) / 10)
    return jerbs_indeed

def linkedin_scraper(job_title, clicks_linkedin):
    job_title = job_title.replace(" ", "%20")
    url_linkedin = f"https://www.linkedin.com/jobs/search?keywords={job_title}&location=Toronto%2C%20Ontario%2C%20Canada&trk=homepage-jobseeker_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0"
    browser.get(url_linkedin)

    try:
        browser.implicitly_wait(random.randint(random.randint(10, 15), random.randint(16, 20))) # seconds
        for _ in range(clicks_linkedin):
            browser.find_element_by_class_name("see-more-jobs").click()
            browser.implicitly_wait(random.randint(random.randint(10, 15), random.randint(16, 20))) # seconds
    except:
        pass

    html_linkedin = browser.page_source
    soup_linkedin = Soup(html_linkedin)

    jerbs_linkedin = []
    for i in range(len(soup_linkedin.find("li", {"class": "result-card job-result-card result-card--with-hover-state"}, mode="all"))):
        list_2 = {}
        list_2["title"] = soup_linkedin.find("li", {"class": "result-card job-result-card result-card--with-hover-state"}, mode="all")[i].find("h3", {"class":"result-card__title job-result-card__title"}).text
        list_2["company"] = soup_linkedin.find("li", {"class": "result-card job-result-card result-card--with-hover-state"}, mode="all")[i].find("h4", {"class":"result-card__subtitle job-result-card__subtitle"}).text
        list_2["url"] = soup_linkedin.find("li", {"class": "result-card job-result-card result-card--with-hover-state"}, mode="all")[i].find("a", {"class": "result-card__full-card-link"}, mode="all")[0].attrs["href"]
        browser.implicitly_wait(random.randint(random.randint(1, 5), random.randint(6, 10))) # seconds
#         time.sleep(random.randint(1, 10) / 8)
        html_jerb = get(list_2["url"])
        soup_jerb = Soup(html_jerb)
        list_2["text"] = BeautifulSoup(" ".join([i.html for i in soup_jerb.find("div", {"class":"description__text description__text--rich"}, mode="all")[0].find("li", mode="all")])).get_text(" ").replace("\n", " ")
        jerbs_linkedin.append(list_2)
        browser.implicitly_wait(random.randint(random.randint(1, 5), random.randint(6, 10))) # seconds
#         time.sleep(random.randint(1, 10) / 8)
    return jerbs_linkedin

###

#-------- ROUTES GO HERE -----------#

@app.route('/cloud')
def start():
    return render_template('cloud.html')

@app.route('/cloud', methods=['GET', 'POST'])
def cloud():
    if request.method == 'POST':
        result = request.form

    try:
        job_title = result["title"]
        jobs_per_page = 5
        search_radius = 100
        # pages = 0
        pages = [0, 10] # list to iterate through
        pg = []
        for page in pages:
            scraped = indeed_scraper(job_title, jobs_per_page, search_radius, page)
            pg += scraped
        df_indeed = pd.DataFrame(pg)
        # clean up duplicates
        df_indeed.drop_duplicates(subset=["title", "company", "text"], keep="first", inplace=True)
        df_indeed.reset_index(inplace=True, drop=True)
    except:
        pass

    try:
        clicks_linkedin = 2
        linkedin_jerbs = linkedin_scraper(job_title, clicks_linkedin)
        df_linkedin = pd.DataFrame(linkedin_jerbs)
        df_linkedin.drop_duplicates(subset=["title", "company", "text"], keep="first", inplace=True)
        df_linkedin.reset_index(inplace=True, drop=True)
    except:
        pass

    df = pd.concat([df_indeed, df_linkedin], axis=0)
    df.drop_duplicates(subset=["title", "company", "text"], keep="first", inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    corpus1 = cleaner(df["text"])
    long_string = ','.join(corpus1)# Create a WordCloud object
    cloud = WordCloud(background_color="white", max_words=5000, contour_width=3, scale=5, contour_color='steelblue')# Generate a word cloud
    cloud.generate(long_string)# Visualize the word cloud
    # plt.figure(figsize=(15,15))
    cloud.to_image()
    cloud.to_file("static/cloud1.png")
    cloud_c = 1

    return render_template('cloud.html', cloud=cloud_c, no_cache=time.time())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = request.form

    doc = cleaner([result["description"]])
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform(doc))

    ### RESULTS ###
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    nkeywords = int(result["nkeywords"])
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,nkeywords)

    # Declare your table
    class ItemTable(Table):
        name = Col('Keyword')
        description = Col('Modelled Occurence Frequency')

    # Get some objects
    class Item(object):
        def __init__(self, name, description):
            self.name = name
            self.description = description
    items = [Item(k,round(keywords[k]*100,2)) for k in keywords]
    prediction = ItemTable(items) # table formatted with Flask_table and rendered direct to html

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    '''Connects to the server'''

    app.run(port=5000, debug=True)
