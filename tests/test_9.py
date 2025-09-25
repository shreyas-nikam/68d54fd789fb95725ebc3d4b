import pytest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Keep the definition_59e3ba83d516411cb58a22eb49a6ff87 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_59e3ba83d516411cb58a22eb49a6ff87 import plot_pair_plot

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'Student_ID': range(5),
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SES_Level': ['High', 'Medium', 'Low', 'High', 'Medium'],
        'Admission_Exam_Score': [80, 75, 60, 90, 70],
        'Clinical_Rotation_Grade': [85, 80, 65, 92, 72],
        'Medical_School_Performance': [1, 0, 0, 1, 0]
    })

def test_plot_pair_plot_valid_input(mocker, sample_dataframe):
    """
    Test with a valid DataFrame and existing hue attribute.
    Ensures seaborn.pairplot and matplotlib.pyplot.show are called.
    """
    mock_pairplot = mocker.patch('seaborn.pairplot')
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')
    
    hue_attribute = 'Gender'
    plot_pair_plot(sample_dataframe, hue_attribute)
    
    mock_pairplot.assert_called_once_with(sample_dataframe, hue=hue_attribute, palette=mocker.ANY)
    mock_plt_show.assert_called_once()

def test_plot_pair_plot_hue_attribute_not_in_df(mocker, sample_dataframe):
    """
    Test when the hue_attribute is not present in the DataFrame columns.
    Expects an appropriate error from seaborn (KeyError or ValueError).
    """
    # Mock seaborn.pairplot to raise an error as it would when hue column is missing
    mock_pairplot = mocker.patch('seaborn.pairplot', side_effect=KeyError("Column 'NonExistent' not found."))
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')

    hue_attribute = 'NonExistent'
    
    with pytest.raises(KeyError) as excinfo:
        plot_pair_plot(sample_dataframe, hue_attribute)
    
    mock_pairplot.assert_called_once_with(sample_dataframe, hue=hue_attribute, palette=mocker.ANY)
    mock_plt_show.assert_not_called()

def test_plot_pair_plot_empty_dataframe(mocker):
    """
    Test with an empty DataFrame.
    Expects an appropriate error, as pairplot cannot plot empty data.
    """
    mock_pairplot = mocker.patch('seaborn.pairplot', side_effect=ValueError("No valid data for plotting."))
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')
    
    empty_df = pd.DataFrame(columns=['col1', 'col2', 'hue_col'])
    hue_attribute = 'hue_col'
    
    with pytest.raises(ValueError) as excinfo:
        plot_pair_plot(empty_df, hue_attribute)
        
    mock_pairplot.assert_called_once_with(empty_df, hue=hue_attribute, palette=mocker.ANY)
    mock_plt_show.assert_not_called()

def test_plot_pair_plot_non_dataframe_input_for_df(mocker):
    """
    Test with a non-DataFrame object for the 'df' argument.
    Expects a TypeError or AttributeError from pandas/seaborn internal checks.
    """
    # We don't mock pairplot's side_effect here directly, as the error might come
    # from initial type checking within plot_pair_plot or seaborn.
    mock_pairplot = mocker.patch('seaborn.pairplot')
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')
    
    invalid_df_input = [1, 2, 3] # A list instead of DataFrame
    hue_attribute = 'Gender'
    
    with pytest.raises((TypeError, AttributeError)): 
        plot_pair_plot(invalid_df_input, hue_attribute)
        
    mock_pairplot.assert_not_called() 
    mock_plt_show.assert_not_called()

def test_plot_pair_plot_hue_attribute_is_none(mocker, sample_dataframe):
    """
    Test with hue_attribute set to None.
    seaborn.pairplot should handle this by not coloring points by hue.
    """
    mock_pairplot = mocker.patch('seaborn.pairplot')
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')
    
    hue_attribute = None # Valid for pairplot, means no hue coloring
    plot_pair_plot(sample_dataframe, hue_attribute)
    
    mock_pairplot.assert_called_once_with(sample_dataframe, hue=hue_attribute, palette=mocker.ANY)
    mock_plt_show.assert_called_once()