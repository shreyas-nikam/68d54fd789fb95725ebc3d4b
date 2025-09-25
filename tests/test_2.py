import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from aif360.datasets import StandardDataset

# This is a placeholder for your actual module import
from definition_e84e882e5e774c2da51d8652e141c701 import preprocess_data

# Helper function to create a dummy DataFrame for testing
def create_dummy_df(num_samples=100, include_sensitive=True, include_numerical=True):
    data = {
        'Student_ID': range(num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'SES_Level': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'Medical_School_Performance': np.random.choice([0, 1], num_samples), # 0: Below_Average, 1: Above_Average
    }
    if include_numerical:
        data['Admission_Exam_Score'] = np.random.rand(num_samples) * 100
        data['Interview_Score'] = np.random.rand(num_samples) * 10
    
    df = pd.DataFrame(data)
    return df

# Define common parameters for tests
SENSITIVE_ATTR_NAMES = ['Gender', 'SES_Level']
TARGET_LABEL_NAME = 'Medical_School_Performance'
FAVORABLE_LABEL = 1
PROTECTED_ATTR_MAP = {
    'Gender': {'privileged_groups': [['Male']], 'unprivileged_groups': [['Female']]},
    'SES_Level': {'privileged_groups': [['High']], 'unprivileged_groups': [['Low'], ['Medium']]}
}

# Test Case 1: Standard functionality with typical data
def test_preprocess_data_standard_case():
    df = create_dummy_df(num_samples=100)
    
    # Expected splits (80/20)
    num_samples = len(df)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size

    X_train, y_train, X_test, y_test, dataset_train_aif, dataset_test_aif = preprocess_data(
        df.copy(), # Pass a copy to prevent modification of original df
        SENSITIVE_ATTR_NAMES,
        TARGET_LABEL_NAME,
        FAVORABLE_LABEL,
        PROTECTED_ATTR_MAP
    )

    # Assert types of returned objects
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)
    assert isinstance(dataset_train_aif, StandardDataset)
    assert isinstance(dataset_test_aif, StandardDataset)

    # Assert shapes based on 80/20 split
    assert X_train.shape[0] == train_size
    assert y_train.shape[0] == train_size
    assert X_test.shape[0] == test_size
    assert y_test.shape[0] == test_size
    
    # Verify AIF360 StandardDataset setup
    assert dataset_train_aif.protected_attribute_names == SENSITIVE_ATTR_NAMES
    assert dataset_test_aif.protected_attribute_names == SENSITIVE_ATTR_NAMES
    assert all(col in X_train.columns for col in ['Gender', 'SES_Level', 'Admission_Exam_Score', 'Interview_Score'])
    # After preprocessing, all feature columns in X_train/X_test should be numeric
    assert X_train.select_dtypes(include=np.number).shape[1] == X_train.shape[1]
    assert X_test.select_dtypes(include=np.number).shape[1] == X_test.shape[1]

# Test Case 2: Empty DataFrame input
def test_preprocess_data_empty_df():
    df_empty = pd.DataFrame(columns=[
        'Student_ID', 'Gender', 'SES_Level', 'Admission_Exam_Score',
        'Interview_Score', 'Medical_School_Performance'
    ])

    X_train, y_train, X_test, y_test, dataset_train_aif, dataset_test_aif = preprocess_data(
        df_empty,
        SENSITIVE_ATTR_NAMES,
        TARGET_LABEL_NAME,
        FAVORABLE_LABEL,
        PROTECTED_ATTR_MAP
    )

    # All returned DataFrames/Series should be empty
    assert X_train.empty
    assert y_train.empty
    assert X_test.empty
    assert y_test.empty
    
    # AIF360 datasets should also be empty
    assert dataset_train_aif.features.empty
    assert dataset_test_aif.features.empty
    
    # Ensure column structure is maintained for empty DataFrames
    expected_cols = [col for col in df_empty.columns if col not in ['Student_ID', TARGET_LABEL_NAME]]
    assert all(col in X_train.columns for col in expected_cols)
    assert y_train.name == TARGET_LABEL_NAME

# Test Case 3: No sensitive attributes provided
def test_preprocess_data_no_sensitive_attributes():
    df = create_dummy_df(num_samples=50)
    
    # Define empty sensitive attributes and protected map
    empty_sensitive_names = []
    empty_protected_map = {}

    X_train, y_train, X_test, y_test, dataset_train_aif, dataset_test_aif = preprocess_data(
        df.copy(),
        empty_sensitive_names,
        TARGET_LABEL_NAME,
        FAVORABLE_LABEL,
        empty_protected_map
    )

    # Basic type and non-empty shape assertions still hold
    assert isinstance(X_train, pd.DataFrame)
    assert X_train.shape[0] > 0
    assert isinstance(dataset_train_aif, StandardDataset)

    # Crucially, AIF360 datasets should reflect no sensitive attributes
    assert dataset_train_aif.protected_attribute_names == []
    assert dataset_train_aif.privileged_groups == []
    assert dataset_train_aif.unprivileged_groups == []
    
    assert dataset_test_aif.protected_attribute_names == []
    assert dataset_test_aif.privileged_groups == []
    assert dataset_test_aif.unprivileged_groups == []

# Test Case 4: Invalid DataFrame type for input 'df'
@pytest.mark.parametrize("invalid_df_input", [
    None,
    [1, 2, 3],
    "not_a_dataframe",
    123
])
def test_preprocess_data_invalid_df_type(invalid_df_input):
    with pytest.raises((TypeError, AttributeError)): # Expect TypeError for non-DataFrame or AttributeError from pandas methods
        preprocess_data(
            invalid_df_input,
            SENSITIVE_ATTR_NAMES,
            TARGET_LABEL_NAME,
            FAVORABLE_LABEL,
            PROTECTED_ATTR_MAP
        )

# Test Case 5: Missing target label or sensitive attributes in DataFrame
def test_preprocess_data_missing_columns():
    df_no_target = create_dummy_df().drop(columns=[TARGET_LABEL_NAME])
    df_no_gender = create_dummy_df().drop(columns=['Gender']) # Drop one of the sensitive columns

    # Missing target label
    with pytest.raises(KeyError, match=TARGET_LABEL_NAME):
        preprocess_data(
            df_no_target,
            SENSITIVE_ATTR_NAMES,
            TARGET_LABEL_NAME,
            FAVORABLE_LABEL,
            PROTECTED_ATTR_MAP
        )

    # Missing one sensitive attribute ('Gender')
    with pytest.raises(KeyError, match='Gender'):
        preprocess_data(
            df_no_gender,
            SENSITIVE_ATTR_NAMES,
            TARGET_LABEL_NAME,
            FAVORABLE_LABEL,
            PROTECTED_ATTR_MAP
        )