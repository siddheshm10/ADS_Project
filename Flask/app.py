from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask import Flask, redirect, render_template, request, url_for
import pickle

app = Flask(__name__)

# Load dataset
zomato_df = pd.read_csv(r'C:\Users\ABC\Desktop\Applied Data Science\ADS-Restaurant-recommendation-System-main\Flask\restaurant.csv')

def get_recommendations(restaurant_name):
    # Case-insensitive and trimmed match
    match = zomato_df[zomato_df['name'].str.lower().str.strip() == restaurant_name.lower().strip()]

    if match.empty:
        return None  # No match found

    input_restaurant = match.iloc[0]

    first_cuisine_keyword = input_restaurant['cuisines'].split()[0]

    similar_restaurants = zomato_df[zomato_df['cuisines'].apply(lambda x: x.split()[0] == first_cuisine_keyword)]

    top_restaurants = similar_restaurants.sort_values(by='Mean Rating', ascending=False)
    top_restaurants = top_restaurants.drop_duplicates(subset=['name', 'cuisines', 'cost'], keep='first')

    return top_restaurants.head(10)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    return render_template('recommend.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        restaurant_name = request.form.get('restaurant_name')
        if not restaurant_name:
            return render_template('result.html', recommended_restaurants=[], error="Please enter a restaurant name.")

        top_restaurants = get_recommendations(restaurant_name)

        if top_restaurants is None:
            return render_template('result.html', recommended_restaurants=[], error="No restaurant found. Please check spelling or try another.")

        top_restaurants_list = top_restaurants.to_dict('records')
        return render_template('result.html', recommended_restaurants=top_restaurants_list)
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
