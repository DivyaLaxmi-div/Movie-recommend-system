import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('movies.csv')

# Fill NaN values
selected_features = ['genres', 'keywords', 'tagline', 'title', 'cast', 'director']
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# Combine features into a single string
combined_text = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['title'] + ' ' + df['cast'] + ' ' + df['director']

# Vectorize the text
vectorizer = TfidfVectorizer()
vectorized_form = vectorizer.fit_transform(combined_text)

# Compute similarity
similarity = cosine_similarity(vectorized_form)

# UI using Streamlit
st.title('🎬 Movie Recommendation System')
st.write("Select a movie and get recommendations based on similar content (genre, cast, keywords, etc.)")

movie_list = df['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button('Recommend'):
    # Find closest matching movie
    find_match_movie = difflib.get_close_matches(selected_movie, movie_list)
    if find_match_movie:
        match_movie = find_match_movie[0]
        index_of_movie = df[df.title == match_movie].index[0]

        # Get similarity scores
        similarity_score = list(enumerate(similarity[index_of_movie]))
        sorted_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader("🎥 Recommended Movies:")
        for i, movie in enumerate(sorted_list[1:11], start=1):
            index = movie[0]
            title = df.iloc[index]['title']
            homepage = df.iloc[index]['homepage']

            if pd.notna(homepage) and homepage.strip() != "":
                st.markdown(f"**🎬 {title}** — [Visit Homepage]({homepage})")
            else:
                st.markdown(f"**🎬 {title}** — Homepage not available")
    else:
        st.error("Movie not found! Try checking the spelling.")
