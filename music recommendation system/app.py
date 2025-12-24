import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Spotify API credentials
CLIENT_ID = "1ff3b09fe3a847098d42b1e07ce0a12d"
CLIENT_SECRET = "df84523cda324cdea4d879ff0e5e6626"

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def content_based_recommendations(song_title, music_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(music_df['text'])  
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(music_df.index, index=music_df['song']).drop_duplicates()
    
    if song_title not in indices:
        st.error(f"Song '{song_title}' not found in the dataset.")
        return [], []
    
    idx = indices[song_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    song_indices = [i[0] for i in sim_scores]
    
    return music_df['song'].iloc[song_indices].tolist(), song_indices

def collaborative_filtering_recommendations(song_similarity_matrix, song_index):
    sim_scores = list(enumerate(song_similarity_matrix[song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    song_indices = [i[0] for i in sim_scores]
    
    return song_indices

def ensemble_recommendations(song_title, music_df, song_similarity_matrix):
    content_recs, content_indices = content_based_recommendations(song_title, music_df)
    if not content_recs:
        return [], []

    song_index = music_df[music_df['song'] == song_title].index[0]
    collab_indices = collaborative_filtering_recommendations(song_similarity_matrix, song_index)
    collab_recs = music_df['song'].iloc[collab_indices].tolist()
    
    combined_recs = list(set(content_recs + collab_recs))[:5]
    combined_posters = [get_song_album_cover_url(song, music_df[music_df['song'] == song]['artist'].values[0])
                        for song in combined_recs]
    
    return combined_recs, combined_posters

st.header('Music Recommender System')

try:
    music = pickle.load(open('df.pkl', 'rb'))
    song_similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Required data files not found. Please ensure df.pkl and similarity.pkl are in the correct location.")
    st.stop()

music_list = music['song'].values
selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = ensemble_recommendations(selected_song, music, song_similarity_matrix)
    if recommended_music_names:
        col1, col2, col3, col4, col5 = st.columns(5)
        for idx, col in enumerate([col1, col2, col3, col4, col5]):
            if idx < len(recommended_music_names):
                col.text(recommended_music_names[idx])
                col.image(recommended_music_posters[idx])
    else:
        st.error("No recommendations found.")
