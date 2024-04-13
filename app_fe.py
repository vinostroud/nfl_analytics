
import streamlit as st
import matplotlib.pyplot as plt
from analytics_py import load_data, get_mean_epa_down1, get_mean_epa_down1and2, get_game_by_game_data, train_regression_model



# Create a Streamlit app with a dropdown bar
st.title("NFL Data and Analysis")

# Prompt the user to select a question
selected_question = st.selectbox("Select a question", [
    "Question 1 - what are the NFL 1st down EPA rankings",
    "Question 2 - what are the NFL 1st and 2nd down EPA rankings",
    "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?",
    "Question 4 - Please show me a simple Regression comparing EPA and Points Scored"
])

# Load data
df_2023 = load_data()

# Check the selected question and display the corresponding data
if selected_question == "Question 1 - what are the NFL 1st down EPA rankings":
    mean_epa_down1 = get_mean_epa_down1(df_2023)
    st.write(mean_epa_down1)
    
elif selected_question == "Question 2 - what are the NFL 1st and 2nd down EPA rankings":
    mean_epa_down1and2 = get_mean_epa_down1and2(df_2023)
    st.write(mean_epa_down1and2)    
    
elif selected_question == "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?":
    df_consolidated_epa_score_combined = get_game_by_game_data(df_2023)
    st.write(df_consolidated_epa_score_combined)
    
elif selected_question == "Question 4 - Please show me a simple Regression comparing EPA and Points Scored":
    sfig = train_regression_model(df_2023)  # Call the function and get the matplotlib figure
    st.pyplot(sfig)  # Display the plot



