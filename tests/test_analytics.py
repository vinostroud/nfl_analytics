import pytest
from src.app_fe import date_selector, load_data, get_mean_epa_down1, get_mean_epa_down1and2, get_game_by_game_data, prepare_data, train_and_plot_regression
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import streamlit as st

import pandas as pd
import pandera as pa
import nfl_data_py as nfl
from pandera import Column, DataFrameSchema
from pandas import DataFrame
from pandas.testing import assert_frame_equal

#Test 1: using pandera to validate the columns I call are in nfl_data_py

df_year = 2023 #I hardcoded this to simplify the test. Perhaps this is not right in context of the test I am running?

def test_validate_columns():
    df = nfl.import_pbp_data([df_year])
    df = df.dropna()
    schema = pa.DataFrameSchema(
        {
            "season_type": Column(str),
            "pass": Column(pa.Float32),
            "rush": Column(pa.Float32),
            "epa": Column(pa.Float32),
            'posteam': Column(str),
            'defteam': Column(str),
        }
    )
    schema.validate(df)
    

# Now that we've established my columns are valid, create sample data to mock the nfl.import_pbp_data function. 
sample_data = pd.DataFrame({
    'season_type': ['REG', 'REG', 'POST'],
    'pass': [1, 0, 1],
    'rush': [0, 1, 0],
    'epa': [0.5, None, 0.2],
    'posteam': ['TeamA', 'TeamB', 'TeamC'],
    'defteam': ['TeamD', None, 'TeamE']
})

#Test 2:  for load_data(df_year: int)
def test_load_data():
    '''Test that load_data returns a DataFrame'''
    df_year = 2015
    with patch('nfl_data_py.import_pbp_data', return_value=sample_data) as mock_import:
        test_df = load_data(df_year)
        assert isinstance(test_df, DataFrame), "The output is not a DataFrame"
        mock_import.assert_called_once_with([df_year])


#Test 3 : For get_mean_epa_down1 - test uses mock data to confirm order is correct

mock_data = pd.DataFrame({
    'down': [1, 1, 1, 1, 1],
    'posteam': ['TeamA', 'TeamA', 'TeamB', 'TeamA', 'TeamB'],
    'epa': [0.5, 0.7, 0.3, 0.6, 0.4]
})

expected_output = pd.DataFrame({
    'Team': ['TeamA', 'TeamB'],
    'First Down EPA': [0.6, 0.35],
    'Rank': [1.0, 2.0]
})

def test_get_mean_epa_down1():
    result = get_mean_epa_down1(mock_data)
    pd.testing.assert_frame_equal(result, expected_output)

'''
Placeholder for Test 4 - getmeanepadown1and2

Placeholder for Test 5 - get_game_by_game_data

Placeholder for Test 6 - repare_data

I don't think a test is needed to plot the chart.
'''