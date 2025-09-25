import pytest
import pandas as pd
# from definition_1826a5342543431c88e0418c2a7bbb24 import calculate_equal_opportunity # Keep this placeholder

@pytest.mark.parametrize("recommendations, sensitive_attribute, true_labels, expected", [
    # Test Case 1: Basic functionality - two groups, non-zero difference
    # Group A: 4 actual positives, 2 recommended (TPR = 0.5)
    # Group B: 4 actual positives, 1 recommended (TPR = 0.25)
    # Expected: abs(0.5 - 0.25) = 0.25
    (
        pd.DataFrame({
            'student_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'course_id': [101, 102, 103, 104, 105, 106, 107, 108],
            'sensitive_attribute': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'is_recommended': [True, False, True, False, True, False, False, False]
        }),
        'sensitive_attribute',
        pd.Series([True, True, True, True, True, True, True, True], index=range(8)),
        0.25
    ),
    # Test Case 2: Perfect Equal Opportunity - two groups, zero difference
    # Group A: 4 actual positives, 2 recommended (TPR = 0.5)
    # Group B: 4 actual positives, 2 recommended (TPR = 0.5)
    # Expected: abs(0.5 - 0.5) = 0.0
    (
        pd.DataFrame({
            'student_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'course_id': [101, 102, 103, 104, 105, 106, 107, 108],
            'sensitive_attribute': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'is_recommended': [True, False, True, False, True, False, True, False]
        }),
        'sensitive_attribute',
        pd.Series([True, True, True, True, True, True, True, True], index=range(8)),
        0.0
    ),
    # Test Case 3: Edge Case - one group has no actual positives (zero denominator for TPR)
    # Group A: TP = 2, Actual Pos = 4 => TPR_A = 0.5
    # Group B: TP = 0, Actual Pos = 0 (all true_labels are False) => TPR_B = 0.0
    # Expected: abs(0.5 - 0.0) = 0.5
    (
        pd.DataFrame({
            'student_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'course_id': [101, 102, 103, 104, 105, 106, 107, 108],
            'sensitive_attribute': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'is_recommended': [True, False, True, False, True, False, True, False]
        }),
        'sensitive_attribute',
        pd.Series([True, True, True, True, False, False, False, False], index=range(8)),
        0.5
    ),
    # Test Case 4: Edge Case - Empty recommendations DataFrame
    # Should result in 0.0 (vacuously fair) as no recommendations to evaluate.
    (
        pd.DataFrame(columns=['student_id', 'course_id', 'sensitive_attribute', 'is_recommended']),
        'sensitive_attribute',
        pd.Series([], dtype=bool),
        0.0
    ),
    # Test Case 5: Error Handling - 'is_recommended' column is missing from recommendations
    # The function expects an 'is_recommended' boolean column to determine recommendations.
    (
        pd.DataFrame({
            'student_id': [1],
            'course_id': [101],
            'sensitive_attribute': ['A'] # Missing 'is_recommended' column
        }),
        'sensitive_attribute',
        pd.Series([True], index=[0]),
        ValueError
    )
])
def test_calculate_equal_opportunity(recommendations, sensitive_attribute, true_labels, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            calculate_equal_opportunity(recommendations, sensitive_attribute, true_labels)
    else:
        result = calculate_equal_opportunity(recommendations, sensitive_attribute, true_labels)
        # Use pytest.approx for floating-point comparisons
        assert result == pytest.approx(expected)