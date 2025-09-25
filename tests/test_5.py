import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from definition_1eecd4411e2543219587047b6fb794d7 import plot_feature_distribution

@pytest.fixture
def mock_plotting(mocker):
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.xlabel')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.legend')
    mocker.patch('matplotlib.pyplot.tight_layout')
    mocker.patch('seaborn.histplot')
    mocker.patch('seaborn.countplot')
    return {
        'plt_show': plt.show,
        'plt_figure': plt.figure,
        'plt_title': plt.title,
        'plt_xlabel': plt.xlabel,
        'plt_ylabel': plt.ylabel,
        'plt_legend': plt.legend,
        'plt_tight_layout': plt.tight_layout,
        'sns_histplot': sns.histplot,
        'sns_countplot': sns.countplot,
    }

df_data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Admission_Exam_Score': [75, 82, 68, 88, 95],
    'SES_Level': ['High', 'Low', 'Medium', 'High', 'Low'],
    'Research_Experience': [1, 0, 1, 1, 0]
})

@pytest.mark.parametrize(
    "df, feature, hue_feature, expected_outcome_or_exception, expected_seaborn_plot_type",
    [
        (df_data, 'Gender', None, None, 'countplot'),
        (df_data, 'Admission_Exam_Score', None, None, 'histplot'),
        (df_data, 'Gender', 'SES_Level', None, 'countplot'),
        (df_data, 'NonExistentFeature', None, KeyError, None),
        ("not_a_dataframe", 'Gender', None, TypeError, None),
    ]
)
def test_plot_feature_distribution(
    mock_plotting, df, feature, hue_feature, expected_outcome_or_exception, expected_seaborn_plot_type
):
    if expected_outcome_or_exception is not None and issubclass(expected_outcome_or_exception, Exception):
        with pytest.raises(expected_outcome_or_exception):
            plot_feature_distribution(df, feature, hue_feature)
        mock_plotting['plt_show'].assert_not_called()
        mock_plotting['plt_figure'].assert_not_called()
        mock_plotting['sns_histplot'].assert_not_called()
        mock_plotting['sns_countplot'].assert_not_called()
    else:
        plot_feature_distribution(df, feature, hue_feature)

        mock_plotting['plt_show'].assert_called_once()
        mock_plotting['plt_figure'].assert_called_once()
        mock_plotting['plt_title'].assert_called_once()
        mock_plotting['plt_xlabel'].assert_called_once()
        mock_plotting['plt_ylabel'].assert_called_once()
        mock_plotting['plt_tight_layout'].assert_called_once()

        if expected_seaborn_plot_type == 'countplot':
            mock_plotting['sns_countplot'].assert_called_once_with(
                data=df, x=feature, hue=hue_feature, palette="viridis"
            )
            mock_plotting['sns_histplot'].assert_not_called()
        elif expected_seaborn_plot_type == 'histplot':
            mock_plotting['sns_histplot'].assert_called_once_with(
                data=df, x=feature, hue=hue_feature, kde=True, palette="viridis"
            )
            mock_plotting['sns_countplot'].assert_not_called()
        else:
            pytest.fail("Expected a seaborn plot type (countplot or histplot) but none was matched.")

        if hue_feature:
            mock_plotting['plt_legend'].assert_called_once()
        else:
            mock_plotting['plt_legend'].assert_not_called()