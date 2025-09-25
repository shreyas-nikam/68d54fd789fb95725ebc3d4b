import pytest
import pandas as pd
from definition_f97d24514b684eb4ba5cc6ebbd1bad52 import generate_synthetic_medical_data

# Define expected column names that the function should return
EXPECTED_COLUMNS = [
    'Student_ID',
    'Gender',
    'SES_Level',
    'Admission_Exam_Score',
    'Interview_Score',
    'Clinical_Rotation_Grade',
    'Research_Experience',
    'Year_of_Admission',
    'Medical_School_Performance'
]

@pytest.mark.parametrize("num_samples, expected_output", [
    # Test case 1: Valid input, typical number of samples
    (100, {'type': pd.DataFrame, 'rows': 100, 'columns': EXPECTED_COLUMNS}),
    # Test case 2: Edge case, zero samples
    (0, {'type': pd.DataFrame, 'rows': 0, 'columns': EXPECTED_COLUMNS}),
    # Test case 3: Edge case, one sample
    (1, {'type': pd.DataFrame, 'rows': 1, 'columns': EXPECTED_COLUMNS}),
    # Test case 4: Invalid input, negative number of samples
    (-5, ValueError),
    # Test case 5: Invalid input, non-integer type for num_samples
    ("invalid_input", TypeError),
])
def test_generate_synthetic_medical_data(num_samples, expected_output):
    """
    Tests the generate_synthetic_medical_data function for various inputs,
    including valid, edge, and invalid cases.
    """
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        # If an exception type is expected, assert that it is raised
        with pytest.raises(expected_output):
            generate_synthetic_medical_data(num_samples)
    else:
        # If a DataFrame is expected, perform detailed checks
        df = generate_synthetic_medical_data(num_samples)

        assert isinstance(df, expected_output['type']), f"Expected return type {expected_output['type']}, but got {type(df)}"
        assert df.shape[0] == expected_output['rows'], f"Expected {expected_output['rows']} rows, but got {df.shape[0]}"
        assert list(df.columns) == expected_output['columns'], f"Expected columns {expected_output['columns']}, but got {list(df.columns)}"

        if num_samples > 0:
            # Additional checks for non-empty dataframes
            # Student_ID should be unique
            assert df['Student_ID'].is_unique, "Student_ID column values are not unique"
            # Medical_School_Performance should be binary (0 or 1)
            assert df['Medical_School_Performance'].isin([0, 1]).all(), "Medical_School_Performance column contains non-binary values"
            # Key sensitive attributes and target should not have null values
            assert not df[['Gender', 'SES_Level', 'Medical_School_Performance']].isnull().any().any(), "Key columns contain null values"