import pytest
import pandas as pd
import numpy as np # For np.nan

from definition_ad7c95bfc5b647ae896f819160431be9 import validate_and_summarize_data

# Helper function to create a basic valid DataFrame for tests
def create_valid_df(num_samples=3):
    data = {
        'Student_ID': list(range(1, num_samples + 1)),
        'Gender': ['Male', 'Female', 'Male'] * (num_samples // 3 + 1),
        'SES_Level': ['High', 'Medium', 'Low'] * (num_samples // 3 + 1),
        'Admission_Exam_Score': [85.5, 92.1, 78.0] * (num_samples // 3 + 1),
        'Interview_Score': [7.0, 8.5, 6.0] * (num_samples // 3 + 1),
        'Clinical_Rotation_Grade': [3.5, 4.0, 3.2] * (num_samples // 3 + 1),
        'Research_Experience': [1, 0, 1] * (num_samples // 3 + 1),
        'Year_of_Admission': [2020, 2021, 2020] * (num_samples // 3 + 1),
        'Medical_School_Performance': [1, 0, 1] * (num_samples // 3 + 1)
    }
    # Trim to num_samples
    for key in data:
        data[key] = data[key][:num_samples]
    
    df = pd.DataFrame(data)
    
    # Explicitly set dtypes for consistency
    df['Student_ID'] = df['Student_ID'].astype(int)
    df['Admission_Exam_Score'] = df['Admission_Exam_Score'].astype(float)
    df['Interview_Score'] = df['Interview_Score'].astype(float)
    df['Clinical_Rotation_Grade'] = df['Clinical_Rotation_Grade'].astype(float)
    df['Research_Experience'] = df['Research_Experience'].astype(int)
    df['Year_of_Admission'] = df['Year_of_Admission'].astype(int)
    df['Medical_School_Performance'] = df['Medical_School_Performance'].astype(int) # Assuming 0/1 for binary
    
    return df

@pytest.fixture
def expected_columns():
    return [
        'Student_ID', 'Gender', 'SES_Level', 'Admission_Exam_Score',
        'Interview_Score', 'Clinical_Rotation_Grade', 'Research_Experience',
        'Year_of_Admission', 'Medical_School_Performance'
    ]

@pytest.fixture
def key_columns_for_missing_values():
    return ['Gender', 'SES_Level', 'Medical_School_Performance']


# Test Case 1: Valid DataFrame (Expected Functionality)
# Checks that a well-formed DataFrame passes validation and prints summary statistics.
def test_valid_dataframe_and_summary_output(capsys, expected_columns, key_columns_for_missing_values):
    df = create_valid_df(num_samples=5) # Use more samples for richer summary stats
    validate_and_summarize_data(df)

    captured = capsys.readouterr() # Capture stdout

    # Check for specific validation success messages
    assert "All required columns are present." in captured.out
    assert "All columns have expected data types." in captured.out
    assert "All Student_ID values are unique." in captured.out
    assert f"No missing values found in key columns: {', '.join(key_columns_for_missing_values)}." in captured.out
    
    # Check for presence of summary statistics
    assert "Summary Statistics for Numeric Columns:" in captured.out
    assert "mean" in captured.out
    assert "std" in captured.out
    assert "Admission_Exam_Score" in captured.out # Example of a numeric column expected to be summarized
    assert "Data validation and summary complete." in captured.out

# Test Case 2: DataFrame with missing required columns
# Verifies that a ValueError is raised if critical columns are absent.
def test_missing_required_column():
    df = create_valid_df()
    df_missing_gender = df.drop(columns=['Gender'])
    missing_col_name = 'Gender'
    with pytest.raises(ValueError, match=f"Expected columns missing: .*\\['{missing_col_name}'\\].*"):
        validate_and_summarize_data(df_missing_gender)

# Test Case 3: DataFrame with incorrect data types for a critical column
# Ensures a ValueError is raised if a column like 'Student_ID' has the wrong data type.
def test_incorrect_data_type():
    df = create_valid_df()
    df_wrong_type = df.copy()
    # Change Student_ID from int to string, which is an incorrect type
    df_wrong_type['Student_ID'] = df_wrong_type['Student_ID'].astype(str)
    with pytest.raises(ValueError, match="Column 'Student_ID' has incorrect data type. Expected type 'int'"):
        validate_and_summarize_data(df_wrong_type)

# Test Case 4: DataFrame with non-unique Student_ID (Primary Key Violation)
# Checks that non-unique 'Student_ID' values lead to a ValueError.
def test_non_unique_student_id():
    df = create_valid_df(num_samples=2)
    df_non_unique_id = df.copy()
    # Make Student_ID '1' duplicated
    df_non_unique_id.loc[1, 'Student_ID'] = 1 
    with pytest.raises(ValueError, match="Duplicate 'Student_ID' values found."):
        validate_and_summarize_data(df_non_unique_id)

# Test Case 5: DataFrame with missing values in a key column ('Gender')
# Confirms that missing values in specified critical columns trigger a ValueError.
def test_missing_values_in_key_column(key_columns_for_missing_values):
    df = create_valid_df()
    df_with_nan = df.copy()
    # Introduce NaN in 'Gender', which is one of the key columns
    df_with_nan.loc[0, 'Gender'] = np.nan 
    with pytest.raises(ValueError, match=f"Missing values found in critical columns: .*\\['Gender'\\].*"):
        validate_and_summarize_data(df_with_nan)
