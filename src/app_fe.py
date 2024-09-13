import streamlit as st

from analytics_py import (
    load_data,
    get_mean_epa_down1,
    get_mean_epa_down1and2,
    get_game_by_game_data,
    prepare_data,
    train_and_plot_regression,
)
import psutil
import os


def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)  # Converts bytes to MB


# Create a Streamlit app with a dropdown bar
st.title("NFL Data and Analysis")


def date_selector():
    st.subheader("Please select a year")
    selected_year = st.selectbox("Select a year", list(range(2010, 2024)))
    try:
        st.write("You selected", selected_year)
        return selected_year
    except (ValueError, NameError):
        st.error(
            "Sorry, the year you selected has no data or is returning an error. Please try a different year."
        )


# Call date_selector function to get the DataFrame for the selected year
df_year = date_selector()


# Prompt the user to select a question
st.subheader("Select a question")
selected_question = st.selectbox(
    "",
    [
        "Q1 - What are the NFL 1st down EPA rankings",
        "Q2 - What are the NFL 1st and 2nd down EPA rankings",
        "Q3 - Show me a game-by-game list of offensive EPA vs turnovers and points scored?",
        "Q4 - Show a simple Regression comparing EPA and Points Scored",
    ],
)

ddf = load_data(df_year)

# Questions that map to functions in analytics_py.py
if selected_question == "Question 1 - what are the NFL 1st down EPA rankings":
    mean_epa_down1 = get_mean_epa_down1(ddf)
    st.write(mean_epa_down1)

elif selected_question == "Question 2 - what are the NFL 1st and 2nd down EPA rankings":
    mean_epa_down1and2 = get_mean_epa_down1and2(ddf)
    st.write(mean_epa_down1and2)


elif (
    selected_question
    == "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?"
):
    df_consolidated_epa_score_combined = get_game_by_game_data(ddf)
    st.write(df_consolidated_epa_score_combined)

elif (
    selected_question
    == "Question 4 - Please show me a simple Regression comparing EPA and Points Scored"
):
    df_consolidated = prepare_data(ddf)
    sfig = train_and_plot_regression(df_consolidated)
    st.pyplot(sfig)


# Leaving this here in case I need to revisit RAM usage. Show memory usage after processing the selected question
# st.write(f"Memory usage after processing: {memory_usage():.2f} MB")
