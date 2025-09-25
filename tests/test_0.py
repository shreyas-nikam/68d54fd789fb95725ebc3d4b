import pytest
import pandas as pd
from definition_3f38a713fb99441c9ae760ea7ae4a3de import generate_synthetic_data

@pytest.mark.parametrize(
    "num_students, num_courses, bias_strength, expected_outcome",
    [
        # Test Case 1: Valid inputs - standard scenario, verify DataFrame properties
        (100, 10, 0.5, {
            "students_rows": 100, "courses_rows": 10, "min_interactions_rows": 1,
            "students_cols": ['student_id', 'gpa', 'major'],
            "courses_cols": ['course_id', 'difficulty', 'subject_area'],
            "interactions_cols": ['student_id', 'course_id', 'interaction_score']
        }),
        
        # Test Case 2: Zero students - should result in empty students_df and interactions_df
        (0, 5, 0.2, {
            "students_rows": 0, "courses_rows": 5, "min_interactions_rows": 0,
            "students_cols": ['student_id', 'gpa', 'major'], # Columns still expected even if empty
            "courses_cols": ['course_id', 'difficulty', 'subject_area'],
            "interactions_cols": ['student_id', 'course_id', 'interaction_score']
        }),
        
        # Test Case 3: Zero courses - should result in empty courses_df and interactions_df
        (10, 0, 0.8, {
            "students_rows": 10, "courses_rows": 0, "min_interactions_rows": 0,
            "students_cols": ['student_id', 'gpa', 'major'],
            "courses_cols": ['course_id', 'difficulty', 'subject_area'], # Columns still expected even if empty
            "interactions_cols": ['student_id', 'course_id', 'interaction_score']
        }),
        
        # Test Case 4: Invalid type for num_students (string instead of int)
        ("invalid", 10, 0.5, TypeError),
        
        # Test Case 5: Negative value for num_students (invalid count)
        (-10, 10, 0.5, ValueError),
    ]
)
def test_generate_synthetic_data(num_students, num_courses, bias_strength, expected_outcome):
    try:
        students_df, courses_df, interactions_df = generate_synthetic_data(
            num_students, num_courses, bias_strength
        )

        # Assertions for successful generation
        assert isinstance(students_df, pd.DataFrame)
        assert isinstance(courses_df, pd.DataFrame)
        assert isinstance(interactions_df, pd.DataFrame)

        assert len(students_df) == expected_outcome["students_rows"]
        assert len(courses_df) == expected_outcome["courses_rows"]
        assert len(interactions_df) >= expected_outcome["min_interactions_rows"]
        
        # Check for expected columns if DataFrame is not empty
        if not students_df.empty:
            for col in expected_outcome["students_cols"]:
                assert col in students_df.columns, f"Missing column '{col}' in students_df"
        # Check for expected columns even if DataFrame is empty, the columns should still be defined
        # For an empty DF, this assertion ensures the schema is correctly initialized.
        elif expected_outcome["students_rows"] == 0 and expected_outcome["students_cols"]:
             assert all(col in students_df.columns for col in expected_outcome["students_cols"]), \
                 f"Expected columns {expected_outcome['students_cols']} not found in empty students_df"


        if not courses_df.empty:
            for col in expected_outcome["courses_cols"]:
                assert col in courses_df.columns, f"Missing column '{col}' in courses_df"
        elif expected_outcome["courses_rows"] == 0 and expected_outcome["courses_cols"]:
            assert all(col in courses_df.columns for col in expected_outcome["courses_cols"]), \
                f"Expected columns {expected_outcome['courses_cols']} not found in empty courses_df"


        if not interactions_df.empty:
            for col in expected_outcome["interactions_cols"]:
                assert col in interactions_df.columns, f"Missing column '{col}' in interactions_df"
        elif expected_outcome["min_interactions_rows"] == 0 and expected_outcome["interactions_cols"]:
             assert all(col in interactions_df.columns for col in expected_outcome["interactions_cols"]), \
                 f"Expected columns {expected_outcome['interactions_cols']} not found in empty interactions_df"


    except Exception as e:
        # Assert that the raised exception is of the expected type
        assert isinstance(e, expected_outcome)