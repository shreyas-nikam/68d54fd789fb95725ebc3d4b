import pytest
import pandas as pd
from unittest.mock import Mock

# Keep the definition_11bdeea33cf64b6c97e5b5dc844b4417 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_11bdeea33cf64b6c97e5b5dc844b4417 import recommend_courses

# --- Mock Model setup ---
class MockModel:
    """
    A mock recommendation model for testing.
    It simulates a model that can predict scores for courses for a given student profile.
    """
    def __init__(self, scores_data=None, exception_on_predict=None):
        self._scores_data = scores_data
        self._exception_on_predict = exception_on_predict
        
        # Default scores if none provided and no exception is expected
        if self._scores_data is None and not exception_on_predict:
            self._scores_data = pd.Series(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                index=[f'course_{i}' for i in range(10)]
            )

    def predict_scores(self, student_profile):
        """
        Simulates the prediction of scores for all courses for a given student profile.
        Raises an exception if configured to do so.
        """
        if self._exception_on_predict:
            raise self._exception_on_predict
        return self._scores_data

# --- Pytest Test Cases ---
@pytest.mark.parametrize(
    "model_instance, student_profile_input, top_n_input, expected_output",
    [
        # Test Case 1: Standard functionality - recommend top N courses correctly
        (
            MockModel(scores_data=pd.Series(
                [0.95, 0.85, 0.75, 0.65, 0.55],
                index=['ML101', 'DL201', 'NLP301', 'CV401', 'Python101']
            )),
            pd.Series({'GPA': 3.5, 'major': 'CS'}, name=1), # Valid student profile (pandas Series)
            3, # Request top 3 courses
            pd.DataFrame({'course_id': ['ML101', 'DL201', 'NLP301'], 'score': [0.95, 0.85, 0.75]})
        ),
        
        # Test Case 2: Edge Case - top_n = 0 should return an empty DataFrame with correct columns
        (
            MockModel(), # Uses default scores
            pd.Series({'GPA': 3.0, 'major': 'EE'}, name=2), # Valid student profile
            0, # Request 0 courses
            pd.DataFrame(columns=['course_id', 'score']) # Expected empty DataFrame
        ),
        
        # Test Case 3: Edge Case - Invalid top_n (negative) should raise ValueError
        (
            MockModel(),
            pd.Series({'GPA': 4.0, 'major': 'Math'}, name=3), # Valid student profile
            -1, # Invalid negative top_n
            ValueError # Expected exception type
        ),
        
        # Test Case 4: Edge Case - Invalid student_profile type should raise TypeError
        (
            MockModel(),
            "invalid_profile_string", # Invalid student profile type (expected pandas.Series or dict)
            5, # Valid top_n
            TypeError # Expected exception type
        ),
        
        # Test Case 5: Edge Case - Model's prediction method raises an internal error, should propagate
        (
            MockModel(exception_on_predict=RuntimeError("Model computation failed")), # Model configured to raise an error
            pd.Series({'GPA': 3.2, 'major': 'Physics'}, name=4), # Valid student profile
            5, # Valid top_n
            RuntimeError # Expected exception type
        ),
    ]
)
def test_recommend_courses(model_instance, student_profile_input, top_n_input, expected_output):
    """
    Tests the recommend_courses function across various valid and edge cases.
    """
    try:
        # Attempt to get recommendations
        actual_output = recommend_courses(model_instance, student_profile_input, top_n_input)
        
        # If no exception occurred, assert the returned DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(actual_output, expected_output, check_dtype=False, check_index_type=False)
        
    except Exception as e:
        # If an exception was raised, assert that its type matches the expected exception type
        assert isinstance(e, expected_output)