# Movie Recommendation System 

This project implements a Machine Learning based Movie Recommendation System that suggests movies based on similarity of genres using TF-IDF vectorization and cosine similarity.

---

## Project Overview

Recommendation systems are widely used in platforms such as Netflix, Amazon Prime, and Spotify.
This project demonstrates a simple **content-based recommendation approach** where movies are recommended based on their genre similarity.

The system analyzes movie metadata and suggests similar movies to the user.

---

## Features

• Content-based movie recommendation
• Uses Machine Learning techniques
• Fast similarity computation using cosine similarity
• Simple and scalable architecture
• Works with small or large datasets

---

## Technologies Used

Python
Pandas
NumPy
Scikit-Learn
TF-IDF Vectorizer

---

## Dataset

The dataset used in this project contains:

* Movie titles
* Movie genres

Example:

| Movie        | Genre            |
| ------------ | ---------------- |
| Avengers     | Action Sci-Fi    |
| Titanic      | Romance Drama    |
| Interstellar | Sci-Fi Adventure |

File used:

movies.csv

---

## Algorithm Used

### 1. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF converts movie genre text into numerical feature vectors so that machine learning algorithms can process them.

Example:

Action Adventure Sci-Fi → vector representation

---

### 2. Cosine Similarity

Cosine similarity measures the similarity between two movies.

Formula:

similarity = (A · B) / (|A| |B|)

Higher similarity score means movies are more related.

---

## System Architecture

Movie Dataset
↓
Data Preprocessing
↓
TF-IDF Vectorization
↓
Cosine Similarity Matrix
↓
Recommendation Engine
↓
Top N Recommended Movies

---

## Example Output

Input Movie:

Avatar

Recommended Movies:

Guardians of the Galaxy
Star Wars
Interstellar
The Martian
Avengers

---

## How to Run the Project

Install required libraries:

pip install pandas scikit-learn numpy

Run the program:

python movie_recommender.py

---

## Future Improvements

• Add user-based collaborative filtering
• Build a web interface using Flask or Streamlit
• Integrate larger datasets (MovieLens dataset)
• Use deep learning models for recommendation

---

## Author

Shreya Sharma
B.Tech – Electronics and Communication Engineering
