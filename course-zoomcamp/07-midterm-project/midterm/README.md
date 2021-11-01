## Timothy Ho's Midterm Project for ML Zoomcamp

### The Problem

Given some very basic high-level features about music, is it possible to predict how popular a song might be? Spotify provides various metrics like loudness, danceability, energy, as well as some more basic metrics like duration and number of sentences in a song. Maybe some of these features could point to the kind of music that ends up being popular. 

### Dataset Used

The dataset that I've chosen to use is from Kaggle:

https://www.kaggle.com/saurabhshahane/spotgen-music-dataset

There are many CSV files, but the ones that I've decided to use are `spotify_tracks.csv` and `lyrics_features.csv`.

### Running the Project

To run the IPython notebook, it is sufficient to download the dataset, put the CSV files into the same directory as the notebook, and run all cells with the same dependences installed as the dependencies required for the ML Zoomcamp.

To run the training script, it is sufficient to download the dataset, put the CSV files into the same directory as the script, and run it with just a few libraries installed:
* numpy
* pandas
* scikit-learn

To run the web service, it is sufficient to run `predict.py`, and send a POST request to the server with JSON data corresponding to what the model expects. This is a combination of track data and lyrics data. This script is almost identical to the one used in a previous homework, down to the port number.
