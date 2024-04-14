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




# Create a Streamlit app with a dropdown bar
st.title("NFL Data and Analysis")

      
def date_selector():
    st.subheader('Please select a year')
    selected_year = st.selectbox('Select a year', list(range(2010, 2024)))
    try:
        st.write('You selected', selected_year)
        df_date = nfl.import_pbp_data([selected_year])
        return df_date
    except ValueError:
        st.error('Sorry, the year you selected has no data or is returning an error. Please try a different year.')
    except NameError:
        st.error('Sorry, the year you selected has no data or is returning an error. Please try a different year.') 

# Call date_selector function to get the DataFrame for the selected year
df_date = date_selector()


# Prompt the user to select a question
st.subheader("Select a question")
selected_question = st.selectbox("", ["Question 1 - what are the NFL 1st down EPA rankings", 
                                                       "Question 2 - what are the NFL 1st and 2nd down EPA rankings", 
                                                       "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?",
                                                       "Question 4 - Please show me a simple Regression comparing EPA and Points Scored"])

# Load data for the specified year (hardcoded as 2023)
#df_2023 = nfl.import_pbp_data([2023])
df_date = df_date[df_date["season_type"] == "REG"]  # Filter to regular season
df_date = df_date[(df_date['pass'] == 1) | (df_date['rush'] == 1)]  # Remove plays with only one team on the field
df_date = df_date.dropna(subset=['epa', 'posteam', 'defteam'])  # Remove plays without EPA, possession team, or defense team

# Check the selected question and display the corresponding data
if selected_question == "Question 1 - what are the NFL 1st down EPA rankings":
    # Calculate mean EPA for down 1
    mean_epa_down1 = df_date[df_date['down'] == 1].groupby('posteam')['epa'].mean().reset_index()
    mean_epa_down1.columns = ['Team', 'First Down EPA']
    mean_epa_down1['Rank'] = mean_epa_down1['First Down EPA'].rank(ascending=False, method='min')
    mean_epa_down1 = mean_epa_down1.sort_values('First Down EPA', ascending=False)
    st.write(mean_epa_down1)
    
elif selected_question == "Question 2 - what are the NFL 1st and 2nd down EPA rankings":
    #mean epa down 1 and 2
    mean_epa_down1and2 = df_date[(df_date['down'].isin([1,2]))].groupby('posteam')['epa'].mean().reset_index().rename(columns = {'posteam':'Team', 'epa' : 'EPA Downs One and Two'}) 
    mean_epa_down1and2['Rank'] = mean_epa_down1and2['EPA Downs One and Two'].rank(ascending=False, method = 'min')
    mean_epa_down1and2[['Rank', 'Team', 'EPA Downs One and Two']]
    mean_epa_down1and2.sort_values('EPA Downs One and Two', ascending = False)
    st.write(mean_epa_down1and2)    
    
elif selected_question == "Question 3 - can you show me a game-by-game list of offensive EPA vs turnovers and points score?":
    # Group by 'game_id', 'home_team', 'posteam', and sum 'fumble_lost' and 'interception'
    grouped_df_home = df_date.groupby(['game_id', 'home_team', 'home_score', 'posteam']).agg({
        'epa': 'mean',
        'fumble_lost': 'sum',
        'interception': 'sum'
    }).reset_index().rename(columns={'game_id': 'Game ID', 'home_team': 'Home Team', 'epa': 'Home Team EPA', 'home_score': 'Home Team Score'})

    # Create a new column 'turnovers' as the sum of 'fumble_lost' and 'interception'
    grouped_df_home['Home Team turnovers'] = grouped_df_home['fumble_lost'] + grouped_df_home['interception']

    # Filter the DataFrame to include only rows where 'home_team' equals 'posteam'
    filtered_df_home = grouped_df_home[grouped_df_home['Home Team'] == grouped_df_home['posteam']].copy()

    # Drop unnecessary columns
    filtered_df_home = filtered_df_home[['Game ID', 'Home Team', 'Home Team EPA', 'Home Team turnovers', 'Home Team Score']]

    # Reset the index after filtering
    filtered_df_home.reset_index(drop=True, inplace=True)

    grouped_df_away = df_date.groupby(['game_id', 'away_team', 'away_score', 'posteam']).agg({
    'epa': 'mean',
    'fumble_lost': 'sum',
    'interception': 'sum'
    }).reset_index().rename(columns={'game_id': 'Game ID', 'away_team': 'Away Team', 'epa': 'Away Team EPA','away_score' : 'Away Team Score'})

    # Create a new column 'turnovers' as the sum of 'fumble_lost' and 'interception'
    grouped_df_away['Away Team turnovers'] = grouped_df_away['fumble_lost'] + grouped_df_away['interception']

    # Filter the DataFrame to include only rows where 'home_team' equals 'posteam'
    filtered_df_away = grouped_df_away[grouped_df_away['Away Team'] == grouped_df_away['posteam']].copy()

    # Drop unnecessary columns
    filtered_df_away = filtered_df_away[['Game ID', 'Away Team', 'Away Team EPA', 'Away Team turnovers', 'Away Team Score']]

    # Reset the index after filtering
    filtered_df_away.reset_index(drop=True, inplace=True)

    #merge home, away
    merged_df_extended = pd.merge(filtered_df_home, filtered_df_away, on='Game ID')
    
    df_consolidated_epa_score_away = filtered_df_away.rename(columns={'game_id': 'Game ID', 'Away Team': 'Team', 'Away Team EPA': 'EPA', 'Away Team turnovers' : 'Turnovers', 'Away Team Score': 'Score'}).reset_index(drop=True)

    df_consolidated_epa_score_home = filtered_df_home.rename(columns={'game_id': 'Game ID', 'Home Team': 'Team', 'Home Team EPA': 'EPA', 'Home Team turnovers' : 'Turnovers', 'Home Team Score': 'Score'}).reset_index(drop=True)

    df_consolidated_epa_score_combined = pd.concat([df_consolidated_epa_score_away, df_consolidated_epa_score_home], axis=0)

    # Reset the index if you want consecutive integers as the index
    df_consolidated_epa_score_combined.reset_index(drop=True, inplace=True)

    st.write(df_consolidated_epa_score_combined)

elif selected_question == "Question 4 - Please show me a simple Regression comparing EPA and Points Scored":
    grouped_df_home = df_date.groupby(['game_id', 'home_team', 'home_score', 'posteam']).agg({
    'epa': 'mean',
    'fumble_lost': 'sum',
    'interception': 'sum'
    }).reset_index().rename(columns={'game_id': 'Game ID', 'home_team': 'Home Team', 'epa': 'Home Team EPA', 'home_score': 'Home Team Score'})

    # Create a new column 'turnovers' as the sum of 'fumble_lost' and 'interception'
    grouped_df_home['Home Team turnovers'] = grouped_df_home['fumble_lost'] + grouped_df_home['interception']

    # Filter the DataFrame to include only rows where 'home_team' equals 'posteam'
    filtered_df_home = grouped_df_home[grouped_df_home['Home Team'] == grouped_df_home['posteam']].copy()

    # Drop unnecessary columns
    filtered_df_home = filtered_df_home[['Game ID', 'Home Team', 'Home Team EPA', 'Home Team turnovers', 'Home Team Score']]

    # Reset the index after filtering
    filtered_df_home.reset_index(drop=True, inplace=True)
    
    grouped_df_away = df_date.groupby(['game_id', 'away_team', 'away_score', 'posteam']).agg({
    'epa': 'mean',
    'fumble_lost': 'sum',
    'interception': 'sum'
    }).reset_index().rename(columns={'game_id': 'Game ID', 'away_team': 'Away Team', 'epa': 'Away Team EPA','away_score' : 'Away Team Score'})

    # Create a new column 'turnovers' as the sum of 'fumble_lost' and 'interception'
    grouped_df_away['Away Team turnovers'] = grouped_df_away['fumble_lost'] + grouped_df_away['interception']

    # Filter the DataFrame to include only rows where 'home_team' equals 'posteam'
    filtered_df_away = grouped_df_away[grouped_df_away['Away Team'] == grouped_df_away['posteam']].copy()

    # Drop unnecessary columns
    filtered_df_away = filtered_df_away[['Game ID', 'Away Team', 'Away Team EPA', 'Away Team turnovers', 'Away Team Score']]

    # Reset the index after filtering
    filtered_df_away.reset_index(drop=True, inplace=True)
    
    df_consolidated_epa_score_away = filtered_df_away.rename(columns={'game_id': 'Game ID', 'Away Team': 'Team', 'Away Team EPA': 'EPA', 'Away Team turnovers' : 'Turnovers', 'Away Team Score': 'Score'}).reset_index(drop=True)

    df_consolidated_epa_score_home = filtered_df_home.rename(columns={'game_id': 'Game ID', 'Home Team': 'Team', 'Home Team EPA': 'EPA', 'Home Team turnovers' : 'Turnovers', 'Home Team Score': 'Score'}).reset_index(drop=True)

    df_consolidated_epa_score_combined = pd.concat([df_consolidated_epa_score_away, df_consolidated_epa_score_home], axis=0)

    # Reset the index if you want consecutive integers as the index
    df_consolidated_epa_score_combined.reset_index(drop=True, inplace=True)


    x_epa = df_consolidated_epa_score_combined['EPA']  # Features
    y_score = df_consolidated_epa_score_combined['Score']  # Target variable  
    
    x_epa_train = x_epa[:-100]
    x_epa_test = x_epa[-400:]

    y_score_train = y_score[:-100]
    y_score_test = y_score[-400:]
    x_epa_train = np.array(x_epa_train).reshape(-1, 1)
    x_epa_test = np.array(x_epa_test).reshape(-1, 1)
    # Train the linear regression model
    reg = LinearRegression()
    reg.fit(x_epa_train, y_score_train)
    train_score = reg.score(x_epa_train, y_score_train)
    test_score = reg.score(x_epa_test, y_score_test)
    score_y_pred = reg.predict(x_epa_test)
    
    sfig, ax = plt.subplots()
    ax.scatter(x_epa_test, y_score_test, color="black", label='Actual')
    ax.plot(x_epa_test, score_y_pred, color="blue", linewidth=3, label='Predicted')

    # Set x and y labels
    ax.set_xlabel('EPA')
    ax.set_ylabel('Score')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    # Add legend
    ax.legend()

    # Show plot
    st.pyplot(sfig)
    