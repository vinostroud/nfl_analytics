import streamlit as st 
import numpy as np
import pathlib
import pandas as pd
import nfl_data_py as nfl
import os
import urllib.request
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from apps.analytics_py import load_data, get_mean_epa_down1, get_mean_epa_down1and2, get_game_by_game_data, prepare_data, train_and_plot_regression   


# Create a Streamlit app with a dropdown bar
st.title("NFL Data and Analysis")

      
def date_selector():
    st.subheader('Please select a year')
    selected_year = st.selectbox('Select a year', list(range(2010, 2024)))
    try:
        st.write('You selected', selected_year)
        return selected_year
    except (ValueError, NameError):
        st.error('Sorry, the year you selected has no data or is returning an error. Please try a different year.')  



# Call date_selector function to get the DataFrame for the selected year
df_year = date_selector()


# Prompt the user to select a question
st.subheader("Select a question")
selected_question = st.selectbox("", ["Question 1 - what are the NFL 1st down EPA rankings", 
                                                       "Question 2 - what are the NFL 1st and 2nd down EPA rankings", 
                                                       "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?",
                                                       "Question 4 - Please show me a simple Regression comparing EPA and Points Scored"])

df = load_data(df_year)

#Questions that map to functions in analytics_py.py
if selected_question == "Question 1 - what are the NFL 1st down EPA rankings":
    mean_epa_down1 = get_mean_epa_down1(df)
    st.write(mean_epa_down1)
        
elif selected_question == "Question 2 - what are the NFL 1st and 2nd down EPA rankings":
    mean_epa_down1and2 = get_mean_epa_down1and2(df)
    st.write(mean_epa_down1and2)   


elif selected_question == "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?":
    df_consolidated_epa_score_combined = get_game_by_game_data(df)
    st.write(df_consolidated_epa_score_combined)

elif selected_question == "Question 4 - Please show me a simple Regression comparing EPA and Points Scored":
    df_consolidated = prepare_data(df)
    sfig = train_and_plot_regression(df_consolidated)
    st.pyplot(sfig)
    
    