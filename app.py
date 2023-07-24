import numpy as np
import pandas as pd
import pickle
import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# laoding models
df = pickle.load(open('df.pkl','rb'))
tfidvector = TfidfVectorizer(analyzer='word',stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)
 

def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)

    return songs

list_songs=np.array(df["song"])
option = st.selectbox(
"Select songs ",
(list_songs))


if st.button('Recommend Me'):
     st.write('songs Recomended for you are:')
     # st.write(movie_recommend(option),show_url(option))
     df = pd.DataFrame({
          'song Recommended': recommendation(option),
     })

     st.table(df)
