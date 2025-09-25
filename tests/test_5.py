import pytest
import pandas as pd
# Keep this block as it is. DO NOT REPLACE or REMOVE.
from definition_133e879f29014f16aae662f5e6a12c05 import calculate_equalized_odds
# End of definition_133e879f29014f16aae662f5e6a12c05 block

@pytest.mark.parametrize("recommendations, sensitive_attribute, true_labels, expected", [
    # Test Case 1: Basic functionality with two sensitive groups and non-zero score.
    # Group M: TPR=0.5, FPR=0.5
    # Group F: TPR=1.0, FPR=0.5
    # Max(|0.5-1.0|, |0.5-0.5|) = max(0.5, 0.0) = 0.5
    (pd.DataFrame({
        'student_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'course_id': [101, 102, 103, 104, 101, 105, 106, 107],
        'sensitive_attribute': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
        'predicted_label': [1, 0, 1, 1, 1, 0, 0, 1]
    }), 'sensitive_attribute', pd.Series([1, 0, 1, 0, 0, 1, 0, 1], name='true_label'), 0.5),

    # Test Case 2: Perfect Equalized Odds (score = 0.0).
    # Group M: TPR=0.5, FPR=0.5
    # Group F: TPR=0.5, FPR=0.5
    # Max(|0.5-0.5|, |0.5-0.5|) = max(0.0, 0.0) = 0.0
    (pd.DataFrame({
        'student_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'course_id': [101, 102, 103, 104, 101, 105, 106, 107],
        'sensitive_attribute': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
        'predicted_label': [1, 0, 1, 0, 0, 1, 1, 0] 
    }), 'sensitive_attribute', pd.Series([1, 0, 1, 0, 0, 1, 0, 1], name='true_label'), 0.0),

    # Test Case 3: Edge case - One group (F) has no relevant items (true_label=1).
    # Group M: TPR=0.5, FPR=0.5
    # Group F: TPR=0.0 (no relevant items), FPR=0.75
    # Max(|0.5-0.0|, |0.5-0.75|) = max(0.5, 0.25) = 0.5
    (pd.DataFrame({
        'student_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'course_id': [101, 102, 103, 104, 101, 105, 106, 107],
        'sensitive_attribute': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
        'predicted_label': [1, 0, 1, 1, 1, 0, 0, 1]
    }), 'sensitive_attribute', pd.Series([1, 0, 0, 0, 0, 1, 0, 0], name='true_label'), 0.5),

    # Test Case 4: Edge case - Single unique sensitive group. Score should be 0 as no disparity can exist.
    (pd.DataFrame({
        'student_id': [1, 1, 3, 3],
        'course_id': [101, 102, 101, 105],
        'sensitive_attribute': ['M', 'M', 'M', 'M'],
        'predicted_label': [1, 0, 1, 0]
    }), 'sensitive_attribute', pd.Series([1, 0, 0, 1], name='true_label'), 0.0),

    # Test Case 5: Error handling - Missing sensitive attribute column in recommendations DataFrame.
    (pd.DataFrame({
        'student_id': [1],
        'course_id': [101],
        'NOT_sensitive_attribute': ['M'], # Incorrect column name
        'predicted_label': [1]
    }), 'sensitive_attribute', pd.Series([1], name='true_label'), KeyError),
])
def test_calculate_equalized_odds(recommendations, sensitive_attribute, true_labels, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            calculate_equalized_odds(recommendations, sensitive_attribute, true_labels)
    else:
        # Using pytest.approx for floating-point comparison
        assert calculate_equalized_odds(recommendations, sensitive_attribute, true_labels) == pytest.approx(expected, abs=1e-9)