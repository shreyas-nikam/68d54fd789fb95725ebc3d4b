import pytest
import pandas as pd

from definition_7ae4a63554a5424e8a44b991331b9865 import visualize_recommendations

# --- Dummy DataFrames for testing ---
df_valid_recommendations = pd.DataFrame([
    {'student_id': 1, 'course_id': 'CS101', 'score': 0.95, 'course_name': 'Intro to CS'},
    {'student_id': 1, 'course_id': 'MA201', 'score': 0.88, 'course_name': 'Calculus II'},
    {'student_id': 2, 'course_id': 'PH101', 'score': 0.92, 'course_name': 'Physics I'},
    {'student_id': 3, 'course_id': 'CS101', 'score': 0.75, 'course_name': 'Intro to CS'},
])

df_empty_recommendations = pd.DataFrame(columns=['student_id', 'course_id', 'score', 'course_name'])


@pytest.mark.parametrize(
    "input_tuple, expected_outcome",
    [
        # Test Case 1: Valid inputs, student found
        # Expects the function to run without error and return None
        ((df_valid_recommendations, 1), None),
        
        # Test Case 2: Valid inputs, student not found
        # Expects the function to handle gracefully (e.g., display nothing) without error and return None
        ((df_valid_recommendations, 99), None),
        
        # Test Case 3: Empty recommendations DataFrame
        # Expects the function to handle gracefully (e.g., display nothing) without error and return None
        ((df_empty_recommendations, 1), None),
        
        # Test Case 4: Invalid 'recommendations' type (not a pandas DataFrame)
        # Expects a TypeError as 'recommendations' must be a DataFrame
        (("not a dataframe", 1), TypeError),
        
        # Test Case 5: Valid 'recommendations' but invalid 'student_id' type (e.g., list instead of int/str)
        # Expects a TypeError as 'student_id' type constraint is violated
        ((df_valid_recommendations, [1]), TypeError),
    ]
)
def test_visualize_recommendations(input_tuple, expected_outcome):
    recommendations_arg, student_id_arg = input_tuple
    try:
        # The function returns None, so we assert for None if no exception is expected
        result = visualize_recommendations(recommendations_arg, student_id_arg)
        assert expected_outcome is None  # No exception expected
        assert result is None  # Function explicitly returns None
    except Exception as e:
        # If an exception was expected, assert that the caught exception is of the expected type
        assert isinstance(e, expected_outcome)