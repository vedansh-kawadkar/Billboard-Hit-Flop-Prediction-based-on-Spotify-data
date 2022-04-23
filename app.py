import numpy as np
import pandas as pd 
import sklearn
import tensorflow as tf 
import streamlit as st 
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Billboard Hit/Flop Predictor", layout="wide")

def load_data(file):
    data = pd.read_csv(file)
    return data

def app():
    st.title('BILLBOARD HIT/FLOP PREDICTOR')
    st.info('Created By: Vedansh K. Kawadkar')

    mkd2 = """
    <table>
        <tr>
            <td><b><i>Danceability</i></b>: Danceability describes how suitable a track is for dancing based on a
                combination
                of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of
                0.0
                is least danceable and 1.0 is most danceable.  
            </td>
            <td><b><i>Energy</i></b>: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of
                intensity
                and activity. Typically, energetic tracks feel fast, loud, and noisy.  
            </td>
            <td><b><i>Key</i></b>: The estimated overall key of the track. Integers map to pitches using standard Pitch
                Class
                notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.  
            </td>
        </tr>
        <tr>
            <td><b><i>Loudness</i></b>: The overall loudness of a track in decibels (dB). Loudness values are averaged
                across
                the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a
                sound that is the primary psychological correlate of physical strength (amplitude).</td>
            <td><b><i>Mode</i></b>: Mode indicates the modality (major or minor) of a track, the type of scale from
                which its
                melodic content is derived. Major is represented by 1 and minor is 0.  
            </td>
            <td><b><i>Speechiness</i></b>: Speechiness detects the presence of spoken words in a track. The more
                exclusively
                speech-like the recording (e.g. talk show), the closer to 1.0 the value.
            </td>
        </tr>
        <tr>
            <td><b><i>Acousticness</i></b>: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0
                represents high confidence the track is acoustic.  
            </td>
            <td><b><i>Instrumentalness</i></b>: Predicts whether a track contains no vocals. The closer the
                instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
            </td>
            <td><b><i>Liveness</i></b>: Detects the presence of an audience in the recording. Higher liveness values
                represent an increased probability that the track was performed live. A value above 0.8 provides strong
                likelihood that the track is live.  
            </td>
        </tr>
        <tr>
            <td><b><i>Valence</i></b>: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a
                track.
                Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low
                valence
                sound more negative (e.g. sad, depressed, angry)  
            </td>
            <td><b><i>Tempo</i></b>: The overall estimated tempo of a track in beats per minute (BPM). In musical
                terminology, tempo is the speed or pace of a given piece and derives directly from the average beat
                duration.  
            </td>
            <td><b><i>Duration_ms</i></b>: The duration of the track in milliseconds.  
            </td>
        </tr>
        <tr>
            <td><b><i>time_signature</i></b>: An estimated overall time signature of a track. The time signature (meter)
                is a
                notational convention to specify how many beats are in each bar (or measure).  
            </td>
            <td><b><i>Chorus Hit</i></b>: This the the authors best estimate of when the chorus would start for the
                track.
                Its the timestamp of the start of the third section of the track (in ms).  
            </td>
            <td><b><i>Sections</i></b>: The number of sections the particular track has. This feature was extracted from
                the
                data recieved by the API call for Audio Analysis of that particular track  
            </td>
        </tr>
    </table>
    """
    #st.markdown(body=mkd, unsafe_allow_html=True)
    with st.beta_expander(label="PROJECT DETAILS", expanded=False):
        st.write("The BILLBOARD HIT/FLOP PREDICTOR was developed as a part of Machine Learning Internship organised by Technocolabs, Indore. The project predicts whether a song with given paramters would be a hit or a flop in a particular decade using a Deep Learning model trained on extensive data consisting of thousands of songs from different decades.")
        st.write("[GitHub Repository](https://github.com/VedanshKumarKawadkar/Billboard-Hit-Flop-Prediction-based-on-Spotify-data)")
    st.write("\n")
    st.write("\n")

    with st.beta_expander(label="PARAMETERS DEFINITION", expanded=False):
        st.write(st.markdown(body=mkd2, unsafe_allow_html=True))
    
    st.write("\n")
    st.write("\n")

    st.success("Set Parameters")
    col1, col3, col2 = st.beta_columns([4.5, 1, 4.5])


    with col1:
        danceability = st.number_input('Enter the DANCEABILITY score. Range:(0 -> 1)', step=0.001, max_value=1.0, min_value=0.0)
        energy = st.number_input('Enter the ENERGY score. Range:(0 -> 1)', step=0.000001, max_value=1.0, min_value=0.0)
        key = st.number_input('Enter the KEY score. Range:(0 - 11)', step=1, value=0, min_value=0, max_value=11 )
        loudness = st.number_input('Enter the LOUDNESS score. Range:(-50 -> 15)', step=0.001, min_value=-50.0, max_value=15.0)
        mode = st.number_input('Enter the MODE score. Range:(0 -> 1)', step=1, value=0, max_value=1, min_value=0)
        speechiness = st.number_input('Enter the SPEECHINESS score. Range:(0 -> 1)', step=0.00001, max_value=1.0, min_value=0.0)
        acousticness = st.number_input('Enter the ACOUSTICNESS score. Range:(0 -> 1)', step=0.000001, max_value=1.0, min_value=0.0)
        instrumentalness = st.number_input('Enter the INSTRUMENTALNESS score. Range:(0 -> 1)', step=0.0001, max_value=1.0, min_value=0.0)

    with col2:    
        liveness = st.number_input('Enter the LIVENESS score. Range:(0 - 1)', step=0.0001, min_value=0.0, max_value=1.0)
        valence = st.number_input('Enter the VALENCE score. Range:(0 - 1)', step=0.001, min_value=0.0, max_value=1.0)
        tempo = st.number_input('Enter the TEMPO score. Range:(0 - 300)', step=0.001, min_value=0.0, max_value=300.0)
        duration_ms = st.number_input('Enter the DURATION_MS score. Range:(10000 - 10000000)', step=1, value=10000, min_value=10000, max_value=10000000)
        time_signature = st.number_input('Enter the TIME SIGNATURE score. Range:(0 - 5)', step=1, value=0, min_value=0, max_value=5)
        chorus_hit = st.number_input('Enter the CHORUS HIT score. Range:(0 - 300)', step=0.00001, min_value=0.0, max_value=300.0)
        sections = st.number_input('Enter the SECTIONS score. Range:(1 - 200)', step=1, value=1, min_value=1, max_value=200)
        
    sample_data = [danceability, energy, key, loudness,
       mode, speechiness, acousticness, instrumentalness, liveness,
       valence, tempo, duration_ms, time_signature, chorus_hit,    
       sections]

    prep_data = np.array(sample_data).reshape(1, -1)

    std_scaler = StandardScaler()
    st.write("\n")
    st.write("\n")

    c1, c2, c3 = st.beta_columns([2, 1, 1])
    with c1:
        st.success("Select Decade")
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
