# Lumaa's Recommender Challenge

This is Joshua Zingale's submition for Lumaa's recommender challenge. I used TF-IDF to generate vectors for each movie in the dataset that I used. A search query is vectorized with the same TF-IDF scheme and the nearest neighbors as determined by cosine similarity are returned as being the "closest" to the query.

## Dataset

I used a subset of this [kaggle dataset](https://www.kaggle.com/datasets/utkarshx27/movies-dataset?resource=download). I formed the subset by randomly (with a seed for reproducibility) sampling 500 rows from the dataset. I have included this dataset in "data/movie_dataset.csv". The included Jupyter Notebook and Python script load the data without need of modification.

## Setup
Python version: Python 3.9.6
Run `pip install -r requirements.txt`

## Running
Using Jupyter Lab, you can open "data_challenge.ipynb" to see the code for this submission. In addition, at the bottom of the notebook are various example search queries. A new query may be made by adding a line that reads `search("your search query here")`.

Alternatively, in a termal you can run `python3 recommend.py "your search query"` to get recommendations. Note: this code was copy and pasted from my notebook, meaning that the formatting for this script is not pretty. Please evaluate the organization of the notebook, not of this Python script.


## Results
The easiest way to view multiple tests of the system is to open "data_challenge.pdf", at the bottom of which are various search queries along with the top five matching movies for each.

For convenience, I have included one such test here:

` search("I love thrilling action movies set in space, with a comedic twist.")`

leading to the following output:

```
Title: Zathura: A Space Adventure (661)
Cosine Similarity 0.1374021423871016
Keywords & Genres: adventure house alien giant robot outer space, Family Fantasy Science Fiction Adventure
Overview: After their father is called into work, two young boys, Walter and Danny, are left in the care of their teenage sister, Lisa, and told they must stay inside. Walter and Danny, who anticipate a boring day, are shocked when they begin playing Zathura, a space-themed board game, which they realize has mystical powers when their house is shot into space. With the help of an astronaut, the boys attempt to return home.

Title: Hard Rain (603)
Cosine Similarity 0.1253841897040928
Keywords & Genres: sheriff rain evacuation armored car crook, Thriller
Overview: Get swept up in the action as an armored car driver (Christian Slater) tries to elude a gang of thieves (led by Morgan Freeman) while a flood ravages the countryside. Hard Rain is "a wild, thrilling, chilling action ride" filled with close calls, uncertain loyalties and heart-stopping heroics.

Title: Capricorn One (3668)
Cosine Similarity 0.11277092550372446
Keywords & Genres: helicopter nasa texas spacecraft beguilement, Drama Action Thriller Science Fiction
Overview: In order to protect the reputation of the American space program, a team of scientists stages a phony Mars landing. Willingly participating in the deception are a trio of well-meaning astronauts, who become liabilities when their space capsule is reported lost on re-entry. Now, with the help of a crusading reporter,they must battle a sinister conspiracy that will stop at nothing to keep the truth

Title: Up (66)
Cosine Similarity 0.10420413628128734
Keywords & Genres: age difference central and south america balloon animation floating in the air, Animation Comedy Family Adventure
Overview: Carl Fredricksen spent his entire life dreaming of exploring the globe and experiencing life to its fullest. But at age 78, life seems to have passed him by, until a twist of fate (and a persistent 8-year old Wilderness Explorer named Russell) gives him a new lease on life.

Title: Galaxina (3534)
Cosine Similarity 0.0858089039722489
Keywords & Genres: android harley davidson cryogenics space travel love, Comedy Science Fiction
Overview: Galaxina is a lifelike, voluptuous android who is assigned to oversee the operations of an intergalactic Space Police cruiser captained by incompetent Cornelius Butt. When a mission requires the ship's crew to be placed in suspended animation for decades, Galaxina finds herself alone for many years, developing emotions and falling in love with the ship's pilot, Thor.
```


## Expected Salary

I expect something equating to $30 an hour. Thus with 20 hours a week and four weeks a month, I expect around $2400 per month.