import pytest
import pandas as pd
from unittest.mock import MagicMock

# Keep the definition_c203cbc6981445849792693000fb6306 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_c203cbc6981445849792693000fb6306 import evaluate_model

@pytest.mark.parametrize("model_input, test_data_input, expected_output", [
    # Test Case 1: Valid inputs, expect successful evaluation metrics (dict of floats)
    # The 'model' argument is expected to be a trained model object, mocked here.
    # 'test_data' is a pandas DataFrame with necessary columns for evaluation.
    (MagicMock(spec_set=['predict']), pd.DataFrame({
        'student_id': [1, 2, 3, 4],
        'course_id': [101, 102, 103, 104],
        'true_label': [1, 0, 1, 0],
        'predicted_score': [0.9, 0.2, 0.8, 0.1]
    }), {'accuracy': float, 'precision': float, 'recall': float}),

    # Test Case 2: Empty test_data, expect ValueError as evaluation cannot proceed without data.
    (MagicMock(spec_set=['predict']), pd.DataFrame(columns=[
        'student_id', 'course_id', 'true_label', 'predicted_score'
    ]), ValueError),

    # Test Case 3: Invalid model type (e.g., None instead of a model object), expect TypeError.
    (None, pd.DataFrame({
        'student_id': [1], 'course_id': [101], 'true_label': [1], 'predicted_score': [0.9]
    }), TypeError),

    # Test Case 4: Invalid test_data type (e.g., a list instead of a pandas DataFrame), expect TypeError.
    (MagicMock(spec_set=['predict']), [1, 2, 3], TypeError),

    # Test Case 5: test_data DataFrame missing critical columns for evaluation (e.g., 'true_label', 'predicted_score'), expect ValueError.
    (MagicMock(spec_set=['predict']), pd.DataFrame({
        'student_id': [1, 2, 3, 4], 'course_id': [101, 102, 103, 104]
    }), ValueError),
])
def test_evaluate_model(model_input, test_data_input, expected_output):
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        # If an exception type is expected, assert that the function raises it
        with pytest.raises(expected_output):
            evaluate_model(model_input, test_data_input)
    else:
        # If a dictionary of expected types is provided, assert the return type and content
        result = evaluate_model(model_input, test_data_input)
        assert isinstance(result, dict)
        # Ensure all expected keys are present and no extra keys, and values are of the expected type
        assert set(result.keys()) == set(expected_output.keys())
        for key, value_type in expected_output.items():
            assert isinstance(result[key], value_type)