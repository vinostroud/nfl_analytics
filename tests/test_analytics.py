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
    try:
        schema.validate(df)
    except pa.errors.SchemaError as e:
        pytest.fail(f"Schema validation failed: {e}")
    



@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'season_type': ['REG', 'REG', 'POST'],
        'pass': [1, 0, 1],
        'rush': [0, 1, 0],
        'epa': [0.5, None, 0.2],
        'posteam': ['TeamA', 'TeamB', 'TeamC'],
        'defteam': ['TeamD', None, 'TeamE']
    })

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'down': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'posteam': ['TeamA', 'TeamA', 'TeamB', 'TeamA', 'TeamB', 'TeamA', 'TeamB', 'TeamA', 'TeamB', 'TeamA', 'TeamB'],
        'epa': [0.59, 0.27, 0.39, 0.12, 0.51, 0.51, 0.33, -0.1, 0.54, 0.57, 0.25]
    })

@pytest.fixture
def expected_output_down1():
    return pd.DataFrame({
        'Team': ['TeamB', 'TeamA'],
        'First Down EPA': [0.45, 0.32666666666666666],
        'Rank': [1.0, 2.0]
    })

@pytest.fixture
def expected_output_down1and2():
    return pd.DataFrame({
        'Team': ['TeamB', 'TeamA'],
        'EPA Downs One and Two': [0.404, 0.32666666666666666],
        'Rank': [1.0, 2.0]        
    })


def test_load_data(sample_data):
    '''Test that load_data returns a DataFrame'''
    df_year = 2015
    with patch('nfl_data_py.import_pbp_data', return_value=sample_data) as mock_import:
        test_df = load_data(df_year)
        assert isinstance(test_df, pd.DataFrame), "The output is not a DataFrame"
        mock_import.assert_called_once_with([df_year])

def test_get_mean_epa_down1(mock_data, expected_output_down1):
    result = get_mean_epa_down1(mock_data).sort_values(by='First Down EPA', ascending=False).reset_index(drop=True)
    expected_output_down1 = expected_output_down1.sort_values(by='First Down EPA', ascending=False).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output_down1)


def test_get_mean_epa_down1and2(mock_data, expected_output_down1and2):
    result = get_mean_epa_down1and2(mock_data).sort_values(by='EPA Downs One and Two', ascending=False).reset_index(drop=True)
    expected_output_down1and2 = expected_output_down1and2.sort_values(by='EPA Downs One and Two', ascending=False).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected_output_down1and2)



'''
Placeholder for Test 4 - getmeanepadown1and2

Placeholder for Test 5 - get_game_by_game_data

Placeholder for Test 6 - repare_data

I don't think a test is needed to plot the chart.
'''