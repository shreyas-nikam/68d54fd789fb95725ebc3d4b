import pytest
import pandas as pd
from definition_8d511619e3c6413d909cdb01cd9884b8 import calculate_demographic_parity

@pytest.mark.parametrize("recommendations_data, sensitive_attribute, expected_output", [
    # Test Case 1: Basic functionality with clear disparity.
    # Group A has 50 recommendations, B has 20, C has 30. Total = 100.
    # Proportions: P(A)=0.5, P(B)=0.2, P(C)=0.3.
    # Demographic parity = max(proportions) - min(proportions) = 0.5 - 0.2 = 0.3.
    (pd.DataFrame({'student_id': range(100),
                   'course_id': range(100, 200),
                   'major': ['A']*50 + ['B']*20 + ['C']*30}), 'major', 0.3),

    # Test Case 2: Perfect demographic parity.
    # All groups have an equal proportion of recommendations.
    # Group A: 30 recs, Group B: 30 recs, Group C: 30 recs. Total = 90.
    # Proportions: P(A)=1/3, P(B)=1/3, P(C)=1/3.
    # Demographic parity = 1/3 - 1/3 = 0.0.
    (pd.DataFrame({'student_id': range(90),
                   'course_id': range(100, 190),
                   'major': ['A']*30 + ['B']*30 + ['C']*30}), 'major', 0.0),

    # Test Case 3: Empty recommendations DataFrame.
    # If there are no recommendations, there can be no disparity. Score should be 0.0.
    (pd.DataFrame(columns=['student_id', 'course_id', 'major']), 'major', 0.0),

    # Test Case 4: Sensitive attribute column not found in the DataFrame.
    # This should raise a KeyError as the column does not exist.
    (pd.DataFrame({'student_id': range(10),
                   'course_id': range(100, 110),
                   'gender': ['M']*5 + ['F']*5}), 'major', KeyError),

    # Test Case 5: Sensitive attribute column contains only one unique group.
    # If all recommendations belong to the same group, there are no other groups to compare against.
    # Therefore, the disparity score should be 0.0.
    (pd.DataFrame({'student_id': range(10),
                   'course_id': range(100, 110),
                   'major': ['A']*10}), 'major', 0.0),
])
def test_calculate_demographic_parity(recommendations_data, sensitive_attribute, expected_output):
    try:
        result = calculate_demographic_parity(recommendations_data, sensitive_attribute)
        # Use a small tolerance for floating-point comparisons
        assert abs(result - expected_output) < 1e-9
    except Exception as e:
        assert isinstance(e, expected_output)