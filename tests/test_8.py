import pytest
import matplotlib
# Use the 'Agg' backend to prevent plots from opening during tests
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from definition_9967642a16ff4a34b194f2efb8a7ec24 import plot_fairness_metrics

@pytest.mark.parametrize(
    "metrics_data, title, metric_keys, expected",
    [
        # Test Case 1: Valid and typical input.
        # Expects no exception and a return of None (as it displays a plot).
        (
            {"gender_male": {"SPD": 0.1, "EOD": 0.05}, "gender_female": {"SPD": -0.1, "EOD": -0.05}},
            "Fairness Metrics by Gender",
            ["SPD", "EOD"],
            None
        ),
        # Test Case 2: Empty metrics_data.
        # The function should ideally handle this gracefully by plotting an empty graph,
        # not raising an error.
        (
            {},
            "Empty Metrics",
            ["SPD"],
            None
        ),
        # Test Case 3: Empty metric_keys.
        # The function should plot axes and a title, but no bars. No error expected.
        (
            {"gender_male": {"SPD": 0.1, "EOD": 0.05}},
            "No Metric Keys",
            [],
            None
        ),
        # Test Case 4: Invalid metrics_data type (e.g., list instead of dict).
        # The function should raise a TypeError for invalid input types.
        (
            [{"SPD": 0.1}], # List instead of dict
            "Invalid Data Input",
            ["SPD"],
            TypeError
        ),
        # Test Case 5: Invalid metric_keys type (e.g., string instead of list).
        # The function should raise a TypeError for invalid input types.
        (
            {"gender_male": {"SPD": 0.1}},
            "Invalid Keys Input",
            "SPD", # String instead of list
            TypeError
        ),
    ]
)
def test_plot_fairness_metrics(metrics_data, title, metric_keys, expected):
    """
    Tests the plot_fairness_metrics function for various inputs, including
    valid cases, edge cases (empty data/keys), and invalid input types.
    """
    if expected is None:
        # For valid inputs, expect the function to execute without error and return None.
        assert plot_fairness_metrics(metrics_data, title, metric_keys) is None
    else:
        # For invalid inputs, expect a specific exception to be raised.
        with pytest.raises(expected):
            plot_fairness_metrics(metrics_data, title, metric_keys)