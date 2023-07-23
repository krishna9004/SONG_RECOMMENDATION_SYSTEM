import numpy as np
import pandas as pd
import pickle
import streamlit as st
# laoding models
df = pickle.load(open('df.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))


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
