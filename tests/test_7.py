import pandas as pd
import pytest

# YOUR_MODULE_BLOCK_START
from definition_b041cbcd828a4dcebcf7cc381643fc00 import apply_fairness_constraint
# YOUR_MODULE_BLOCK_END

# Helper function to create a dummy DataFrame for testing
def create_imbalanced_recommendations_df():
    """
    Creates a synthetic pandas DataFrame of recommendations with an inherent bias
    for testing fairness constraints. Group 'A' is set to have generally higher scores.
    """
    data = {
        'student_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'course_id': ['CS101', 'MA201', 'PH101', 'CS102', 'MA202', 'PH201', 'CS103', 'MA301', 'AR101', 'BI101'],
        'score': [0.95, 0.90, 0.88, 0.85, 0.82, 0.79, 0.75, 0.70, 0.65, 0.60], # Scores indicating preference
        'sensitive_group': ['A', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'A', 'B'] # Initial imbalance: Group A gets higher scores overall
    }
    df = pd.DataFrame(data)
    return df

@pytest.mark.parametrize(
    "recommendations_input, sensitive_attribute, constraint_type, strength, expected",
    [
        # Test Case 1: Expected functionality - positive strength, demographic_parity
        # Expectation: Returns an adjusted DataFrame. As exact scores are unknown,
        # 'DataFrameAdjusted' is a sentinel to trigger custom assertion logic.
        (create_imbalanced_recommendations_df(), 'sensitive_group', 'demographic_parity', 0.5, "DataFrameAdjusted"),
        
        # Test Case 2: Edge Case - Zero strength
        # Expectation: Returns a DataFrame identical to the input, as no adjustment should occur.
        # 'expected' is the exact original DataFrame for direct comparison.
        (create_imbalanced_recommendations_df(), 'sensitive_group', 'demographic_parity', 0.0, create_imbalanced_recommendations_df()),
        
        # Test Case 3: Edge Case - Invalid sensitive_attribute column
        # Expectation: Raises a KeyError (likely if pandas column access fails).
        (create_imbalanced_recommendations_df(), 'non_existent_group', 'demographic_parity', 0.5, KeyError),
        
        # Test Case 4: Edge Case - Invalid constraint_type
        # Expectation: Raises a ValueError (for unsupported constraint types).
        (create_imbalanced_recommendations_df(), 'sensitive_group', 'unsupported_constraint', 0.5, ValueError),
        
        # Test Case 5: Edge Case - recommendations is not a pandas DataFrame
        # Expectation: Raises a TypeError.
        (None, 'sensitive_group', 'demographic_parity', 0.5, TypeError),
    ]
)
def test_apply_fairness_constraint(recommendations_input, sensitive_attribute, constraint_type, strength, expected):
    # Make a deep copy of the input DataFrame if it's a DataFrame,
    # to preserve its original state for comparison after function call.
    original_df_for_comparison = None
    if isinstance(recommendations_input, pd.DataFrame):
        original_df_for_comparison = recommendations_input.copy(deep=True)

    try:
        # Call the function under test
        result = apply_fairness_constraint(recommendations_input, sensitive_attribute, constraint_type, strength)

        if expected == "DataFrameAdjusted":
            # For this special case, we assert that a DataFrame is returned,
            # and that its content (e.g., scores/rankings) has been adjusted (i.e., not identical to original).
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_index_equal(result.columns, original_df_for_comparison.columns)
            assert result.shape == original_df_for_comparison.shape
            assert not result.equals(original_df_for_comparison), "DataFrame should have been adjusted but is identical."
        elif isinstance(expected, pd.DataFrame):
            # For cases where an exact DataFrame is expected (e.g., strength=0),
            # use pandas testing function for exact DataFrame comparison.
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, expected, check_exact=True)
        else:
            # If 'expected' is not a known success type (e.g., a string or an unexpected type),
            # then the test should fail as this indicates an unhandled scenario.
            pytest.fail(f"Unexpected successful outcome. Expected: {expected}")
            
    except Exception as e:
        # If an exception is expected (i.e., 'expected' is an Exception class),
        # assert that the raised exception is of the expected type.
        assert isinstance(e, expected), f"Expected {expected.__name__} but got {type(e).__name__}"