#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender System Challenge
# 
# In this notebook, I, Joshua Zingale, create a movie recommender system with a subset of a [kaggle dataset](https://www.kaggle.com/datasets/utkarshx27/movies-dataset?resource=download), which I have also included in this repository.

# ## Data Loading and Cleaning

# In[1]:


import pandas as pd
pd.options.mode.copy_on_write = True

# Load only smaller subset of the movies per the instructions
df = pd.read_csv("data/movie_dataset.csv").sample(500, random_state = 115)

# View a small random sample to get a feel for the data
df.sample(3, random_state = 935)


# In[2]:


df.columns


# ### Choosing Columns
# Looking at the column names and some examples, the most immediately relevant columns for this challenge seem to be "genres", "keywords", "original_title", and "overview", though other columns could be used to improve relevancy. Also, I am going to filter the data only to include those movies originally in English.

# In[3]:


# Load the most relevant columns for all movies in the English language, dropping any rows with NaN values
df = df[df["original_language"] == "en"][["index", "genres", "keywords", "original_title", "overview"]].dropna()

print(len(df))


# In[4]:


df.sample(3, random_state=115)


# ### Combining Columns
# Fortunately, most of the data are English movies so I retained enough data. To finish cleaning the data, I now will concatenate genres, the keywords, and the film overview into a single column, which will later be turned into a vector.

# In[5]:


# Build a new DataFrame with a composite "desription" column
df["description"] = df["original_title"] + " " + df["genres"] + " " + df["keywords"] + " " + df["overview"]


# In[6]:


df.sample(3, random_state = 935)


# ## Vectorized Movie Storage
# 
# I created a vectorized database wherein each movie has an associated vector. A movie's vector is a TF-IDF vector of the text in the composite "description" field created above.
# 
# To generate a TF-IDF vector for a piece of text, I implemented a tokenizer. The tokenizer removes punctuation, sets everything to lowercase, and then attempts to get the stem of each word, e.g. by removing verb endings or plural markers. Once the text is in a normilized ("stemified") form, each word is a token; and a vector is asigned to the text based on the frequency of words present in the text and based on the inverse document frequency of each token.
# 
# I downloaded a list of English stop words from [here](https://gist.github.com/larsyencken/1440509), which I used to remove stop words from the descriptions during the tokenization stage. I added "movie" and "movies" to the list of stop words because queries of the form "I like moves that..." were resulting in irrelevant movies simply for a match to "movie".
# 
# When vectorizing a query, any word following I is removed to prevent "love" in constructions like "I love horrific war films" from biasing the results toward romantic comedies.
# 

# In[7]:


from nltk.stem import PorterStemmer
import numpy as np
ps = PorterStemmer()


# In[8]:


# Load in the stopwords list
with open("stopwords.txt") as f:
    stopwords = f.read().split()
stopwords = set(stopwords)


# In[9]:


class Tokenizer():
    def __init__(self, documents):
        """
        Initializes a tokenizer for a set of documents.
        The vocabulary for the Tokenizer is determined by the documents used to initialize it.
        """

        ## Get the vocabulary
        self.vocabulary = set()
        for text in documents:
            self.vocabulary = self.vocabulary.union(set(self._stemify(text)))

        self.vocabulary_size = len(self.vocabulary)
        

        ## Get the stem to token id mappings
        self.stem_to_id = dict()
        self.id_to_stem = dict()

        for i, word in enumerate(self.vocabulary):
            self.stem_to_id[word] = i
            self.id_to_stem[i] = word

        ## Get the inverse document frequencies for each token
        self.idf = np.zeros(self.vocabulary_size)
        for text in documents:
            self.idf += self.frequency_vectorize(text).clip(0, 1)

        self.idf = np.log(len(documents)/self.idf)

    def tokenize(self, text: str) -> list:
        """Tokenizes input text"""
        return [self.stem_to_id[stem] for stem in self._stemify(text) if stem in self.vocabulary]

    def vectorize(self, text: str, smoothing = 0.0) -> np.ndarray:
        """Returns a tf-idf vector for the input text, where each index contains the tf-idf of a token in the input text.
        smoothing is the amount of smoothing added to the vector."""
        vec = np.zeros(self.vocabulary_size) + smoothing
        for token_id in self.tokenize(text):
            vec[token_id] += 1
        return vec * self.idf
    
    def frequency_vectorize(self, text: str) -> np.ndarray:
        """Returns a frequency vector for the input text, where each index contains the number of appearances of a token in the input text."""
        vec = np.zeros(self.vocabulary_size)
        for token_id in self.tokenize(text):
            vec[token_id] += 1
        return vec
        
    def _remove_punctuation(self, text: str) -> str:
        """ Removes common punctuation from a string """
        for mark in ["!", "(", ")", ";", ":", "\"", ",", ".", "?"]:
            text = text.replace(mark, "")
        return text
    
    def _stemify(self, text: str) -> list:
        """ Converts text into a list of stems without punctuation"""
        text = self._remove_punctuation(text)
        words = text.lower().split()
        words = [ps.stem(word) for word in words if word not in stopwords]
        return words


# In[10]:


class VectorDB():
    """
    Stores documents in a vector database, wherein lookups use a vector similarity metric, i.e. cosin similarity.
    """
    def __init__(self, data, embedded_row, embedding_function):
        """
        Initializes a vector database for a set of data.

        :param data: pandas DataFrame around which this database wraps 
        :embedding_function: function that takes a document to an embedding thereof
        """

        ## Build the vector database
        embedding_size = embedding_function(data[embedded_row].iloc[0]).size
        
        
        self.db = np.ndarray((len(data), embedding_size))
        self.data = data
        
        for i, document in enumerate(data[embedded_row]):
            self.db[i] = embedding_function(document)

        # normalize each db row
        self.db /= np.linalg.norm(self.db, axis = 1, keepdims = True)

    def search(self, x, k = 1, return_similarities = False):
        """Returns the top k closest matches for input vector x"""
        # normalize x
        x = x / np.linalg.norm(x)

        # Get top k
        scores = self.db @ x
        top_idc = np.argpartition(scores, -k)[-k:]

        # Sort top k
        top_idc = sorted(top_idc, key = lambda i: -scores[i])
        if return_similarities:
            return self.data.iloc[top_idc], scores[top_idc]
        return self.data.iloc[top_idc]


# In[11]:


# Get a Tokenizer for the data
tokenizer = Tokenizer(df.loc[:, "description"])

print(f"The vocabulary has {tokenizer.vocabulary_size} words")


# In[12]:


# Create the vectorized database
db = VectorDB(df, embedded_row = "description", embedding_function = tokenizer.vectorize)


# In[13]:


def vectorize_query(text):
    """Vectorizes a search query"""

    words = text.lower().split()
    new_words = [words[0]]
    # Remove any word the follows "I"
    for prev_word, word in zip(words[:-1], words[1:]):
        if prev_word != "i":
            new_words.append(word)

    text = " ".join(new_words)
    
    return tokenizer.vectorize(text)


# ## Testing

# In[14]:


def search(text, k = 5):
    rows, scores = db.search(vectorize_query(text), k = k, return_similarities = True)

    print("Results the following query:", text)
    for row, score in zip(rows.iloc, scores):
        title = row["original_title"]
        index = row["index"]
        overview = row["overview"]
        keywords = row["keywords"]
        genres = row["genres"]
        print(f"Title: {title} ({index})\nCosine Similarity {score}\nKeywords & Genres: {keywords}, {genres}\nOverview: {overview}\n")


# In[15]:


search("I love thrilling action movies set in space, with a comedic twist.")


# In[16]:


search("I like action movies set in space")


# In[17]:


search("I like movies that are informative and teach me something")


# In[18]:


search("I like calm documentaries about nature.")


# In[19]:


search("I like calm documentaries about war.")


# In[20]:


search("I love comedies for the family")


# In[21]:


search("I love comedies")


# In[22]:


search("I like horror films that take place in the wild")

