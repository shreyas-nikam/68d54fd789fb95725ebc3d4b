import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Keep the `your_module` block as it is. DO NOT REPLACE or REMOVE the block.
from definition_1441341f522a4d118d9d390cf9b4e91a import plot_fairness_metrics_comparison 


@pytest.fixture
def mock_plot(mocker):
    """Fixture to mock matplotlib.pyplot and seaborn plotting functions."""
    # Mock matplotlib.pyplot functions
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.subplots', return_value=(mocker.MagicMock(), mocker.MagicMock()))
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.xlabel')
    mocker.patch('matplotlib.pyplot.ylabel')
    mocker.patch('matplotlib.pyplot.legend')
    mocker.patch('matplotlib.pyplot.tight_layout')
    mocker.patch('matplotlib.pyplot.close')
    
    # Mock seaborn functions
    mocker.patch('seaborn.set_palette')
    mock_barplot = mocker.patch('seaborn.barplot')

    # Return the mocked seaborn.barplot for specific assertions
    return mock_barplot, mocker.patch('seaborn.set_palette')


@pytest.mark.parametrize("baseline_metrics, mitigated_metrics, metric_name, sensitive_attribute, expected_exception", [
    # Test Case 1: Standard comparison with valid data
    (
        {
            'Gender': {'Statistical Parity Difference': 0.1, 'Equal Opportunity Difference': 0.05},
            'SES_Level': {'Statistical Parity Difference': 0.15, 'Equal Opportunity Difference': 0.07}
        },
        {
            'Gender': {'Statistical Parity Difference': 0.02, 'Equal Opportunity Difference': 0.01},
            'SES_Level': {'Statistical Parity Difference': 0.03, 'Equal Opportunity Difference': 0.02}
        },
        'Statistical Parity Difference',
        'Gender',
        None
    ),
    # Test Case 2: Metric name not found in dictionaries
    (
        {
            'Gender': {'Statistical Parity Difference': 0.1},
        },
        {
            'Gender': {'Statistical Parity Difference': 0.02},
        },
        'NonExistentMetric', # This metric is not defined in the input dictionaries
        'Gender',
        KeyError
    ),
    # Test Case 3: Sensitive attribute not found in dictionaries
    (
        {
            'Gender': {'Statistical Parity Difference': 0.1},
        },
        {
            'Gender': {'Statistical Parity Difference': 0.02},
        },
        'Statistical Parity Difference',
        'NonExistentAttribute', # This attribute is not defined in the input dictionaries
        KeyError
    ),
    # Test Case 4: Empty metrics dictionaries (leads to KeyError when trying to access sensitive_attribute)
    (
        {}, # Empty baseline_metrics
        {}, # Empty mitigated_metrics
        'Statistical Parity Difference',
        'Gender',
        KeyError 
    ),
    # Test Case 5: Invalid input type for metrics dictionary (e.g., string instead of dict)
    (
        "not_a_dict", # Invalid type for baseline_metrics
        {
            'Gender': {'Statistical Parity Difference': 0.02},
        },
        'Statistical Parity Difference',
        'Gender',
        TypeError # Expected to fail when trying to access 'Gender' on a string
    ),
])
def test_plot_fairness_metrics_comparison(mock_plot, baseline_metrics, mitigated_metrics, metric_name, sensitive_attribute, expected_exception):
    mock_barplot, mock_set_palette = mock_plot

    if expected_exception:
        with pytest.raises(expected_exception):
            plot_fairness_metrics_comparison(baseline_metrics, mitigated_metrics, metric_name, sensitive_attribute)
    else:
        plot_fairness_metrics_comparison(baseline_metrics, mitigated_metrics, metric_name, sensitive_attribute)
        
        # Assert that plotting functions were called for successful cases
        mock_barplot.assert_called_once()
        mock_set_palette.assert_called_once_with("viridis") # As per notebook spec for color palette

        # Assert matplotlib.pyplot functions were called
        plt.figure.assert_called_once()
        plt.subplots.assert_called_once()
        plt.title.assert_called_once()
        plt.xlabel.assert_called_once()
        plt.ylabel.assert_called_once()
        plt.legend.assert_called_once()
        plt.tight_layout.assert_called_once()
        plt.close.assert_called_once()
        plt.show.assert_called_once() # Ensure show is called to display plot