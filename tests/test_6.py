import pytest
import pandas as pd
import numpy as np

# definition_ccb89ea63d6f40d4917a96846980a428 block - DO NOT REPLACE OR REMOVE
from definition_ccb89ea63d6f40d4917a96846980a428 import calculate_predictive_parity
# End of definition_ccb89ea63d6f40d4917a96846980a428 block

def test_perfect_predictive_parity():
    """
    Test case for perfect predictive parity where all sensitive groups have the same PPV.
    Expected disparity score: 0.0
    """
    recommendations = pd.DataFrame({
        'sensitive_attribute': ['A', 'A', 'B', 'B', 'A', 'B'],
        'predicted_label': [1, 1, 1, 1, 0, 0]
    })
    # Group A: 2 positive predictions (indices 0, 1).
    # Group B: 2 positive predictions (indices 2, 3).
    true_labels = pd.Series([1, 1, 1, 1, 0, 0])
    # For Group A's positive predictions (indices 0, 1): true_labels[0]=1, true_labels[1]=1. PPV_A = 2/2 = 1.0
    # For Group B's positive predictions (indices 2, 3): true_labels[2]=1, true_labels[3]=1. PPV_B = 2/2 = 1.0
    # Expected difference = abs(1.0 - 1.0) = 0.0
    expected_score = 0.0
    assert calculate_predictive_parity(recommendations, 'sensitive_attribute', true_labels) == pytest.approx(expected_score)

def test_moderate_predictive_parity():
    """
    Test case for a scenario with clear disparity between sensitive groups' PPVs.
    """
    recommendations = pd.DataFrame({
        'sensitive_attribute': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'predicted_label': [1, 1, 1, 1, 1, 1, 1, 1] # All are positive predictions
    })
    true_labels = pd.Series([1, 1, 0, 0, 1, 1, 1, 0])
    # Group A: 4 positive predictions (indices 0,1,2,3). True positives at 0, 1. PPV_A = 2/4 = 0.5
    # Group B: 4 positive predictions (indices 4,5,6,7). True positives at 4, 5, 6. PPV_B = 3/4 = 0.75
    # Expected difference = abs(0.5 - 0.75) = 0.25
    expected_score = 0.25
    assert calculate_predictive_parity(recommendations, 'sensitive_attribute', true_labels) == pytest.approx(expected_score)

def test_empty_recommendations():
    """
    Test case for an empty recommendations DataFrame.
    Expected disparity score: 0.0 (no predictions, thus no disparity)
    """
    recommendations = pd.DataFrame(columns=['sensitive_attribute', 'predicted_label'])
    true_labels = pd.Series([])
    expected_score = 0.0
    assert calculate_predictive_parity(recommendations, 'sensitive_attribute', true_labels) == pytest.approx(expected_score)

def test_missing_sensitive_attribute_column():
    """
    Test case for when the specified sensitive attribute column is missing from the recommendations DataFrame.
    Expected: KeyError
    """
    recommendations = pd.DataFrame({
        'student_id': [1, 2, 3],
        'predicted_label': [1, 0, 1]
    })
    true_labels = pd.Series([1, 0, 1])
    # Expect KeyError because 'non_existent_attribute' is not a column
    with pytest.raises(KeyError):
        calculate_predictive_parity(recommendations, 'non_existent_attribute', true_labels)

def test_group_with_no_positive_predictions():
    """
    Test case where one sensitive group has no positive predictions.
    PPV for such a group should typically be considered 0.0 for disparity calculation.
    """
    recommendations = pd.DataFrame({
        'sensitive_attribute': ['A', 'A', 'A', 'B', 'B', 'B'],
        'predicted_label': [1, 1, 0, 0, 0, 0] # Group A has 2 positive preds, Group B has 0.
    })
    true_labels = pd.Series([1, 0, 0, 0, 0, 0])
    # Group A: Positive predictions at indices 0, 1. true_labels[0]=1, true_labels[1]=0. PPV_A = 1/2 = 0.5
    # Group B: No positive predictions. PPV_B = 0.0 (assuming this handling)
    # Expected difference = abs(0.5 - 0.0) = 0.5
    expected_score = 0.5
    assert calculate_predictive_parity(recommendations, 'sensitive_attribute', true_labels) == pytest.approx(expected_score)