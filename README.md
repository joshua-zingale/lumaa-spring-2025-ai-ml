# Lumaa's Recommender Challenge

This is Joshua Zingale's submition for Lumaa's recommender challenge. I used TF-IDF to generate vectors for each movie in the dataset that I used. A search query is vectorized with the same TF-IDF scheme and the nearest neighbors as determined by cosine similarity are returned as being the "closest" to the query.

## Dataset

I used a subset of this [kaggle dataset](https://www.kaggle.com/datasets/utkarshx27/movies-dataset?resource=download). I formed the subset by randomly (with a seed for reproducibility) sampling 500 rows from the dataset. I have included this dataset in "data/movie_dataset.csv". The included Jupyter Notebook and Python script load the data without need of modification.

## Setup
Python version: Python 3.9.6
Run `pip install -r requirements.txt`