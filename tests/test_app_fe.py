import pytest
from src.app_fe import date_selector
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import streamlit as st

def test_date_selector():
    with patch('streamlit.selectbox') as mock_selectbox:
        mock_selectbox.return_value = 2021
        selected_year = date_selector()
        assert selected_year == 2021
        mock_selectbox.assert_called_once_with('Select a year', list(range(2010, 2024)))