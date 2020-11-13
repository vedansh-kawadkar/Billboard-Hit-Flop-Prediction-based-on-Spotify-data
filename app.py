import numpy as np
import pandas as pd 
import sklearn
import tensorflow as tf 
import streamlit as st 
from sklearn.preprocessing import StandardScaler

def load_data(file):
    data = pd.read_csv(file)
    return data

def app():
    st.header('SPOTIFY HIT PREDICTOR')
    st.info('Created By: Vedansh K. Kawadkar')
    st.selectbox('About The Parameters:', [
        'Danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.',
        'Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. ',
        'Key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.',
        'Loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. ',
        'Mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.',
        'Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. ',
        'Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.',
        'Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.',
        'Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.',
        'Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)',
        'Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. ',
        'Duration_ms: 	The duration of the track in milliseconds.',
        'time_signature: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).',
        'Chorus_hit: This the the authors best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track (in milliseconds). This feature was extracted from the data recieved by the API call for Audio Analysis of that particular track.',
        'Sections: The number of sections the particular track has. This feature was extracted from the data recieved by the API call for Audio Analysis of that particular track.'
    
    ])
    danceability = st.number_input('Enter the DANCEABILITY score. Range:(0-1)', step=0.001)
    energy = st.number_input('Enter the ENERGY score. Range:(0 - 1)', step=0.000001)
    key = st.number_input('Enter the KEY score. Range:(0 - 11)', step=1, value=0 )
    loudness = st.number_input('Enter the LOUDNESS score. Range:(-50 - 15)', step=0.001)
    mode = st.number_input('Enter the MODE score. Range:(0 - 1)', step=1, value=0)
    speechiness = st.number_input('Enter the SPEECHINESS score. Range:(0 - 1)', step=0.00001)
    acousticness = st.number_input('Enter the ACOUSTICNESS score. Range:(0 - 1)', step=0.000001)
    instrumentalness = st.number_input('Enter the INSTRUMENTALNESS score. Range:(0 - 1)', step=0.0001)
    liveness = st.number_input('Enter the LIVENESS score. Range:(0 - 1)', step=0.0001)
    valence = st.number_input('Enter the VALENCE score. Range:(0 - 1)', step=0.001)
    tempo = st.number_input('Enter the TEMPO score. Range:(0 - 300)', step=0.001)
    duration_ms = st.number_input('Enter the DURATION_MS score. Range:(10000 - 10000000)', step=1, value=0)
    time_signature = st.number_input('Enter the TIME SIGNATURE score. Range:(0 - 5)', step=1, value=0)
    chorus_hit = st.number_input('Enter the CHORUS HIT score. Range:(0 - 300)', step=0.00001)
    sections = st.number_input('Enter the SECTIONS score. Range:(1 - 200)', step=1, value=0)
    
    sample_data = [danceability, energy, key, loudness,
       mode, speechiness, acousticness, instrumentalness, liveness,
       valence, tempo, duration_ms, time_signature, chorus_hit,    
       sections]

    prep_data = np.array(sample_data).reshape(1, -1)

    std_scaler = StandardScaler()
    
    decade = st.selectbox('Choose the decade for which you want to check the trend.',['1960s', '1970s', '1980s', '1990s', '2000s', '2010s'] )

    if st.button('PREDICT'):
        if decade == '1960s':
            predictor = tf.keras.models.load_model('Trained_model_60')
            prediction = predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)
      
        elif decade == '1970s':
            predictor = tf.keras.models.load_model('Trained_model_70')
            prediction= predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)

        elif decade == '1980s':
            predictor = tf.keras.models.load_model('Trained_model_80')
            prediction = predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)
            
        elif decade == '1990s':
            predictor = tf.keras.models.load_model('Trained_model_90')
            prediction = predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)
            
        if decade == '2000s':
            predictor = tf.keras.models.load_model('Trained_model_00')
            prediction = predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)
            
        elif decade == '2010s':
            predictor = tf.keras.models.load_model('Trained_model_10')
            prediction = predictor.predict(std_scaler.fit_transform(prep_data))
            result = np.argmax(prediction)
            
        if result == 0:
            st.error('The song with above parameters has a high chance of being FLOP.')
        else:
            st.success('The song with above parameters has a high chance of being HIT.')

if __name__ == '__main__':
    app()
