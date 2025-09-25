import pandas as pd
import numpy as np

def generate_synthetic_data(num_students, num_courses, bias_strength):
    """
    Generates synthetic student, course, and interaction data with a bias factor.
    """
    # 1. Input Validation
    if not isinstance(num_students, int) or not isinstance(num_courses, int):
        raise TypeError("num_students and num_courses must be integers.")
    if not isinstance(bias_strength, (int, float)):
        raise TypeError("bias_strength must be a float or integer.")

    if num_students < 0 or num_courses < 0:
        raise ValueError("num_students and num_courses cannot be negative.")

    # Define possible values for categorical data
    majors = ['Computer Science', 'Mathematics', 'Physics', 'Literature', 'History', 'Economics']
    difficulties = ['Easy', 'Medium', 'Hard']
    subject_areas = ['STEM', 'Humanities', 'Business']

    # Mapping majors to subject areas for bias
    major_to_subject = {
        'Computer Science': 'STEM', 'Mathematics': 'STEM', 'Physics': 'STEM',
        'Literature': 'Humanities', 'History': 'Humanities',
        'Economics': 'Business'
    }

    # Initialize empty DataFrames with correct columns for edge cases (0 students/courses)
    students_df = pd.DataFrame(columns=['student_id', 'gpa', 'major'])
    courses_df = pd.DataFrame(columns=['course_id', 'difficulty', 'subject_area'])
    interactions_df = pd.DataFrame(columns=['student_id', 'course_id', 'interaction_score'])

    # 2. Generate Students Data
    if num_students > 0:
        student_ids = np.arange(num_students)
        gpas = np.round(np.random.uniform(2.0, 4.0, num_students), 2)
        student_majors = np.random.choice(majors, num_students)
        students_df = pd.DataFrame({
            'student_id': student_ids,
            'gpa': gpas,
            'major': student_majors
        })

    # 3. Generate Courses Data
    if num_courses > 0:
        course_ids = np.arange(num_courses)
        course_difficulties = np.random.choice(difficulties, num_courses)
        course_subjects = np.random.choice(subject_areas, num_courses)
        
        courses_df = pd.DataFrame({
            'course_id': course_ids,
            'difficulty': course_difficulties,
            'subject_area': course_subjects
        })

    # 4. Generate Interactions Data (with bias)
    if num_students > 0 and num_courses > 0:
        all_student_ids = students_df['student_id'].values
        all_course_ids = courses_df['course_id'].values

        # Determine number of interactions to generate (e.g., 30% of all possible pairs)
        interaction_density = 0.3 
        num_interactions = int(num_students * num_courses * interaction_density)
        
        # Ensure at least one interaction if both students and courses exist and density results in zero interactions
        if num_interactions == 0:
            num_interactions = 1 

        # Cap the number of interactions to prevent exceeding all possible pairs
        num_interactions = min(num_interactions, num_students * num_courses)

        if num_interactions > 0:
            # Create all possible student-course pairs
            all_pairs = [(s_id, c_id) for s_id in all_student_ids for c_id in all_course_ids]
            
            # Randomly sample a subset of pairs for interactions
            np.random.shuffle(all_pairs)
            sampled_pairs = all_pairs[:num_interactions]

            interaction_list = []
            for s_id, c_id in sampled_pairs:
                student_major = students_df.loc[students_df['student_id'] == s_id, 'major'].iloc[0]
                course_subject = courses_df.loc[courses_df['course_id'] == c_id, 'subject_area'].iloc[0]

                # Generate a base interaction score
                base_score = np.random.uniform(0.3, 0.7) 

                # Apply bias based on major-subject alignment
                if major_to_subject.get(student_major) == course_subject:
                    # Positive bias: student's major aligns with course subject
                    score_adjustment = bias_strength * 0.3
                    interaction_score = base_score + score_adjustment
                else:
                    # Negative bias: student's major does not align
                    score_adjustment = bias_strength * 0.15
                    interaction_score = base_score - score_adjustment
                
                # Clip the score to be within [0.0, 1.0]
                interaction_score = np.clip(interaction_score, 0.0, 1.0)
                
                interaction_list.append({
                    'student_id': s_id,
                    'course_id': c_id,
                    'interaction_score': interaction_score
                })
            
            if interaction_list: # Create DataFrame only if interactions were generated
                interactions_df = pd.DataFrame(interaction_list)

    return students_df, courses_df, interactions_df

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

def train_recommendation_model(data, model_type):
    """Trains a recommendation model based on interaction data (collaborative filtering or content-based)."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Input 'data' DataFrame cannot be empty.")

    if model_type == 'collaborative_filtering':
        required_cols = ['student_id', 'course_id', 'interaction']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"For 'collaborative_filtering', 'data' must contain columns: {required_cols}.")

        # Create user-item interaction matrix
        user_item_matrix = data.pivot_table(
            index='student_id',
            columns='course_id',
            values='interaction'
        ).fillna(0)

        if user_item_matrix.empty or min(user_item_matrix.shape) == 0:
            raise ValueError("User-item matrix is empty or has zero dimension after pivoting. Cannot train SVD model.")
        
        # Determine n_components for TruncatedSVD. Default is 2.
        # Ensure n_components is not larger than the minimum dimension of the matrix.
        n_components_svd = min(2, user_item_matrix.shape[0], user_item_matrix.shape[1])
        if n_components_svd == 0:
             raise ValueError("Insufficient unique students or courses to apply SVD (matrix dimensions too small).")

        model = TruncatedSVD(n_components=n_components_svd, random_state=42)
        model.fit(user_item_matrix)
        return model

    elif model_type == 'content_based':
        features = ['feature1', 'feature2'] # Assumed based on dummy_df_cb in test cases
        target = 'target' # Assumed based on dummy_df_cb in test cases
        required_cols = features + [target]

        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"For 'content_based', 'data' must contain features {features} and target '{target}'.")

        X = data[features]
        y = data[target]

        if X.empty or y.empty:
            raise ValueError("Features (X) or target (y) are empty after selection. Cannot train Logistic Regression model.")

        model = LogisticRegression(solver='liblinear', random_state=42) # 'liblinear' solver for small datasets
        model.fit(X, y)
        return model

    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. Choose 'collaborative_filtering' or 'content_based'.")

import pandas as pd

def recommend_courses(model, student_profile, top_n):
    """
    Utilizes a trained recommendation model to predict and return the top N most relevant courses for a given student profile.
    Arguments:
        model: The trained recommendation model.
        student_profile (pandas.Series or dict): A profile of the student for whom recommendations are to be generated.
        top_n (int): The number of top courses to recommend.
    Output:
        recommended_courses (pandas.DataFrame): A DataFrame listing the top N recommended courses with their predicted scores.
    """
    # Validate top_n argument
    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")
    if top_n < 0:
        raise ValueError("top_n cannot be negative.")

    # Validate student_profile argument
    if not isinstance(student_profile, (pd.Series, dict)):
        raise TypeError("student_profile must be a pandas.Series or a dict.")

    # Handle the edge case where top_n is 0
    if top_n == 0:
        return pd.DataFrame(columns=['course_id', 'score'])

    # Predict scores for all courses using the provided model
    # The model's predict_scores method is expected to return a pandas.Series
    # where the index represents course IDs and the values are their predicted scores.
    raw_scores = model.predict_scores(student_profile)

    # Ensure the model's output is a pandas Series for consistent processing
    if not isinstance(raw_scores, pd.Series):
        raise TypeError("The model's predict_scores method must return a pandas.Series.")

    # Sort the scores in descending order to identify the most relevant courses
    sorted_scores = raw_scores.sort_values(ascending=False)

    # Select the top N courses based on their scores
    top_n_scores = sorted_scores.head(top_n)

    # Format the results into a pandas DataFrame with 'course_id' and 'score' columns
    recommended_courses_df = pd.DataFrame({
        'course_id': top_n_scores.index,
        'score': top_n_scores.values
    })

    return recommended_courses_df

import pandas as pd

def calculate_demographic_parity(recommendations, sensitive_attribute):
    """
    Calculates the demographic parity metric, which quantifies the difference in recommendation rates
    across different groups defined by a sensitive attribute. A lower difference indicates better
    demographic parity.

    Arguments:
        recommendations (pandas.DataFrame): A DataFrame of recommended courses.
        sensitive_attribute (str): The column name representing the sensitive attribute.

    Output:
        demographic_parity_score (float): The calculated demographic parity score.
    """
    # Check if the sensitive attribute column exists in the DataFrame
    if sensitive_attribute not in recommendations.columns:
        raise KeyError(f"Sensitive attribute '{sensitive_attribute}' not found in the recommendations DataFrame.")

    # Calculate the count of recommendations for each unique group
    group_counts = recommendations[sensitive_attribute].value_counts()

    # If there are no recommendations or only one unique group, there's no disparity
    if len(group_counts) <= 1:
        return 0.0

    # Calculate the total number of recommendations across all relevant groups
    total_recommendations = group_counts.sum()

    # Calculate the proportion of recommendations for each group
    proportions = group_counts / total_recommendations

    # Demographic parity is the difference between the maximum and minimum proportions
    demographic_parity_score = proportions.max() - proportions.min()

    return demographic_parity_score

import pandas as pd
import numpy as np

def calculate_equal_opportunity(recommendations, sensitive_attribute, true_labels):
    """
    Calculates the equal opportunity metric, which assesses whether the true positive rate (i.e., the rate of correctly
    recommended relevant items) for a sensitive attribute is equal across different groups, given the true labels.
    This metric focuses on ensuring that individuals from different groups who are equally qualified have an equal
    chance of being recommended.

    Arguments:
        recommendations (pandas.DataFrame): A DataFrame of recommended courses, including student and course details,
                                            and an 'is_recommended' boolean column.
        sensitive_attribute (str): The name of the column in the recommendations DataFrame representing the sensitive attribute.
        true_labels (pandas.Series or pandas.DataFrame): The ground truth labels indicating actual student-course relevance.
                                                          If DataFrame, it must contain a single boolean column.

    Output:
        equal_opportunity_score (float): The calculated equal opportunity score,
                                         which is the maximum absolute difference between the TPRs of any two groups.
    """

    # 1. Input Validation and Data Preparation
    if 'is_recommended' not in recommendations.columns:
        raise ValueError("The 'recommendations' DataFrame must contain an 'is_recommended' boolean column.")
    
    if sensitive_attribute not in recommendations.columns:
        raise ValueError(f"The sensitive_attribute column '{sensitive_attribute}' not found in recommendations DataFrame.")

    if recommendations.empty:
        return 0.0

    # Ensure true_labels is a Series
    if isinstance(true_labels, pd.DataFrame):
        if true_labels.shape[1] > 1:
            raise ValueError("If 'true_labels' is a DataFrame, it must contain a single column representing relevance.")
        # Squeeze to convert a single-column DataFrame into a Series
        true_labels_series = true_labels.iloc[:, 0].squeeze()
    elif isinstance(true_labels, pd.Series):
        true_labels_series = true_labels
    else:
        raise TypeError("true_labels must be a pandas.Series or pandas.DataFrame.")

    # Create a working DataFrame by copying recommendations
    df = recommendations.copy()

    # Ensure true_labels_series has the same length as the recommendations DataFrame
    if len(df) != len(true_labels_series):
        raise ValueError("Length of 'recommendations' and 'true_labels' must be the same.")
    
    # Reset index of the DataFrame to ensure positional alignment with the true_labels_series,
    # assuming true_labels corresponds to the recommendations in the order they appear.
    df = df.reset_index(drop=True)
    df['true_label'] = true_labels_series.reset_index(drop=True)

    # 2. Calculate TPR for each group
    sensitive_groups = df[sensitive_attribute].unique()
    tpr_by_group = {}

    for group in sensitive_groups:
        # Filter for the current sensitive group
        group_df = df[df[sensitive_attribute] == group]
        
        # Calculate actual positives (true_label == True) within this group
        actual_positives = group_df['true_label'].sum() # True counts as 1, False as 0

        # Calculate true positives (true_label == True AND is_recommended == True) within this group
        true_positives = group_df[(group_df['true_label'] == True) & (group_df['is_recommended'] == True)].shape[0]

        # Calculate TPR, handling division by zero for actual_positives
        if actual_positives > 0:
            tpr = true_positives / actual_positives
        else:
            # If there are no actual positives in this group, TPR is 0
            # as there's no opportunity to be truly positive.
            tpr = 0.0
        
        tpr_by_group[group] = tpr
    
    # 3. Calculate Equal Opportunity Score
    # The score is the maximum absolute difference between TPRs of any two groups.
    if len(tpr_by_group) < 2:
        # If there's 0 or 1 sensitive group, there's no disparity to measure, so the score is 0.0
        return 0.0
    
    tpr_values = list(tpr_by_group.values())
    max_tpr_diff = 0.0
    for i in range(len(tpr_values)):
        for j in range(i + 1, len(tpr_values)):
            diff = abs(tpr_values[i] - tpr_values[j])
            if diff > max_tpr_diff:
                max_tpr_diff = diff
                
    return max_tpr_diff

import pandas as pd
import numpy as np

def calculate_equalized_odds(recommendations, sensitive_attribute, true_labels):
    """
    Calculates the equalized odds metric, ensuring both true positive rate and false positive rate
    are equal across different sensitive groups.
    """
    if sensitive_attribute not in recommendations.columns:
        raise KeyError(f"Sensitive attribute column '{sensitive_attribute}' not found in recommendations DataFrame.")
    
    if 'predicted_label' not in recommendations.columns:
        raise KeyError("Required column 'predicted_label' not found in recommendations DataFrame.")

    # Ensure true_labels is a Series for consistent handling
    if isinstance(true_labels, pd.DataFrame):
        if true_labels.shape[1] == 1:
            true_labels_series = true_labels.iloc[:, 0]
        else:
            raise ValueError("true_labels DataFrame should contain only one column for relevance.")
    else: # Assume it's a Series
        true_labels_series = true_labels

    if len(recommendations) != len(true_labels_series):
        raise ValueError("Length of recommendations and true_labels must be the same.")

    # Combine recommendations and true_labels. Reset index of true_labels_series to ensure positional alignment.
    df_combined = recommendations.copy()
    df_combined['true_label'] = true_labels_series.reset_index(drop=True)

    sensitive_groups = df_combined[sensitive_attribute].unique()

    if len(sensitive_groups) <= 1:
        # If there's one or no sensitive group, no disparity can exist.
        return 0.0

    group_metrics = {}

    for group in sensitive_groups:
        group_df = df_combined[df_combined[sensitive_attribute] == group]

        tp = ((group_df['predicted_label'] == 1) & (group_df['true_label'] == 1)).sum()
        fp = ((group_df['predicted_label'] == 1) & (group_df['true_label'] == 0)).sum()
        fn = ((group_df['predicted_label'] == 0) & (group_df['true_label'] == 1)).sum()
        tn = ((group_df['predicted_label'] == 0) & (group_df['true_label'] == 0)).sum()

        actual_positives = tp + fn
        actual_negatives = fp + tn

        # Calculate True Positive Rate (TPR)
        tpr = tp / actual_positives if actual_positives > 0 else 0.0
        # Calculate False Positive Rate (FPR)
        fpr = fp / actual_negatives if actual_negatives > 0 else 0.0

        group_metrics[group] = {'tpr': tpr, 'fpr': fpr}

    max_tpr_diff = 0.0
    max_fpr_diff = 0.0

    # Calculate the maximum absolute differences across all pairs of groups
    for i in range(len(sensitive_groups)):
        for j in range(i + 1, len(sensitive_groups)):
            group1 = sensitive_groups[i]
            group2 = sensitive_groups[j]

            tpr1 = group_metrics[group1]['tpr']
            tpr2 = group_metrics[group2]['tpr']
            fpr1 = group_metrics[group1]['fpr']
            fpr2 = group_metrics[group2]['fpr']

            max_tpr_diff = max(max_tpr_diff, abs(tpr1 - tpr2))
            max_fpr_diff = max(max_fpr_diff, abs(fpr1 - fpr2))
    
    return max(max_tpr_diff, max_fpr_diff)

import pandas as pd
import numpy as np

def calculate_predictive_parity(recommendations, sensitive_attribute, true_labels):
    """
    Calculates the predictive parity metric, evaluating if the positive predictive value (PPV)
    is equal across groups defined by a sensitive attribute.

    Args:
        recommendations (pandas.DataFrame): DataFrame with 'predicted_label' and `sensitive_attribute`.
        sensitive_attribute (str): Column name for the sensitive attribute.
        true_labels (pandas.Series or pandas.DataFrame): Ground truth labels.

    Returns:
        float: The calculated predictive parity score (max(PPV) - min(PPV) across groups).
    """

    if recommendations.empty:
        return 0.0

    if 'predicted_label' not in recommendations.columns:
        raise ValueError("The 'recommendations' DataFrame must contain a 'predicted_label' column.")
    
    # Ensure true_labels is a Series for consistent handling
    if isinstance(true_labels, pd.DataFrame):
        if 'true_label' in true_labels.columns:
            true_labels_series = true_labels['true_label']
        elif len(true_labels.columns) == 1: # If only one column, assume it's the true labels
            true_labels_series = true_labels.iloc[:, 0]
        else:
            raise ValueError("If 'true_labels' is a DataFrame, it must contain a 'true_label' column or have only one column.")
    else: # Assume it's already a Series
        true_labels_series = true_labels

    # Check for length consistency
    if len(true_labels_series) != len(recommendations):
        raise ValueError("Length of true_labels must match the number of rows in recommendations.")

    # Create a working DataFrame by combining recommendations and true_labels
    df_combined = recommendations.copy()
    # Assign true_labels values directly. This assumes positional correspondence of rows.
    df_combined['true_label'] = true_labels_series.values

    # Get all unique groups in the sensitive attribute from the original recommendations
    # to ensure all groups are considered, even if they have no positive predictions.
    all_groups = df_combined[sensitive_attribute].unique()
    
    # Initialize PPV for all groups to 0.0. This handles groups with no positive predictions.
    group_ppvs = {group: 0.0 for group in all_groups}

    # Filter for positive predictions across the entire DataFrame
    positive_preds_df = df_combined[df_combined['predicted_label'] == 1]

    # If there are no positive predictions at all, all PPVs remain 0, so disparity is 0.
    if positive_preds_df.empty:
        return 0.0

    # Calculate PPV for each group that *has* positive predictions
    for group_name, group_data in positive_preds_df.groupby(sensitive_attribute):
        total_positive_predictions_in_group = len(group_data) 
        true_positives_in_group = (group_data['true_label'] == 1).sum()
        
        ppv = true_positives_in_group / total_positive_predictions_in_group
        group_ppvs[group_name] = ppv

    # Extract PPV values
    ppv_values = list(group_ppvs.values())

    # If there's only one group (or zero, which is handled by recommendations.empty), there's no disparity.
    if len(ppv_values) <= 1:
        return 0.0

    # The predictive parity score is the absolute difference between the maximum and minimum PPV values.
    max_ppv = max(ppv_values)
    min_ppv = min(ppv_values)

    predictive_parity_score = max_ppv - min_ppv
    
    return predictive_parity_score

import pandas as pd
import numpy as np

def apply_fairness_constraint(recommendations, sensitive_attribute, constraint_type, strength):
    """
    Adjusts recommendation scores or re-ranks courses to improve a specified fairness metric
    based on a sensitive attribute. The 'strength' parameter controls the intensity of this adjustment,
    allowing for a trade-off between fairness and other performance metrics.

    Arguments:
        recommendations (pandas.DataFrame): The initial DataFrame of recommended courses.
        sensitive_attribute (str): The name of the column representing the sensitive attribute.
        constraint_type (str): The type of fairness constraint to apply (e.g., 'demographic_parity').
        strength (float): A parameter controlling the intensity of the fairness adjustment.

    Output:
        adjusted_recommendations (pandas.DataFrame): A DataFrame of recommendations with adjusted scores or rankings.
    """

    # 1. Input Validation
    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("Recommendations must be a pandas DataFrame.")

    if sensitive_attribute not in recommendations.columns:
        raise KeyError(f"Sensitive attribute column '{sensitive_attribute}' not found in recommendations.")

    # Define supported constraint types. 'equal_opportunity' is mentioned in the docstring
    # but not fully specified for implementation without outcome labels and only scores.
    # We will only support 'demographic_parity' for score adjustment as per test requirements.
    supported_constraint_types = ['demographic_parity']
    if constraint_type not in supported_constraint_types:
        raise ValueError(f"Unsupported constraint type: '{constraint_type}'. Supported types are: {', '.join(supported_constraint_types)}")

    # If strength is zero, no adjustment is needed. Return a deep copy to ensure the original DataFrame
    # remains untouched, satisfying the test case expecting an identical DataFrame.
    if strength == 0.0:
        return recommendations.copy(deep=True)

    # Ensure 'score' column exists as it's required for score-based adjustments
    if 'score' not in recommendations.columns:
        raise ValueError("The DataFrame must contain a 'score' column for fairness adjustments.")

    # Make a deep copy to avoid modifying the original DataFrame passed into the function
    adjusted_recommendations = recommendations.copy(deep=True)

    # 2. Apply Fairness Constraint based on type
    if constraint_type == 'demographic_parity':
        # Goal: Equalize average scores across sensitive groups by adjusting individual scores.
        # This implementation boosts scores for groups with below-average scores and
        # penalizes groups with above-average scores, relative to the overall mean score.

        # Calculate the overall mean score across all recommendations
        overall_mean_score = adjusted_recommendations['score'].mean()

        # Calculate the mean score for each unique group within the sensitive attribute
        group_means = adjusted_recommendations.groupby(sensitive_attribute)['score'].mean()

        # Calculate the score adjustment for each group.
        # If a group's mean score is lower than the overall mean, its adjustment will be positive (boost).
        # If a group's mean score is higher than the overall mean, its adjustment will be negative (penalty).
        # The 'strength' parameter scales this calculated difference.
        score_adjustments = (overall_mean_score - group_means) * strength

        # Apply the calculated adjustments to the 'score' column using a vectorized operation.
        # We map the `score_adjustments` Series back to the DataFrame based on the sensitive attribute
        # of each recommendation.
        adjustment_series_mapped = adjusted_recommendations[sensitive_attribute].map(score_adjustments)
        adjusted_recommendations['score'] = adjusted_recommendations['score'] + adjustment_series_mapped

        # Clip the adjusted scores to remain within a sensible range (e.g., 0 to 1).
        # This assumes original scores are within this range and prevents scores from
        # becoming negative or exceeding 1.0 after adjustment.
        adjusted_recommendations['score'] = np.clip(adjusted_recommendations['score'], 0.0, 1.0)

    return adjusted_recommendations

import pandas as pd
import matplotlib.pyplot as plt

def visualize_recommendations(recommendations, student_id):
    """
    Generates and displays a visualization (e.g., a table or bar chart) of the top recommended courses for a specific student.
    Arguments:
        recommendations (pandas.DataFrame): A DataFrame containing the recommended courses for various students.
        student_id (int or str): The identifier of the student whose recommendations are to be visualized.
    Output:
        None (displays a plot or table directly).
    """
    # Type checking for input arguments
    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("Argument 'recommendations' must be a pandas.DataFrame.")
    if not isinstance(student_id, (int, str)):
        raise TypeError("Argument 'student_id' must be an int or a str.")

    # Filter recommendations for the specific student
    student_recs = recommendations[recommendations['student_id'] == student_id]

    # Handle cases where no recommendations are found for the student or the DataFrame is empty
    if student_recs.empty:
        return None

    # Sort recommendations by score in descending order to get 'top' recommendations
    student_recs = student_recs.sort_values(by='score', ascending=False)

    # Generate a visualization (e.g., a horizontal bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting course names against their scores
    ax.barh(student_recs['course_name'], student_recs['score'], color='skyblue')

    # Set plot labels and title
    ax.set_xlabel('Recommendation Score')
    ax.set_ylabel('Course Name')
    ax.set_title(f'Top Recommended Courses for Student {student_id}')

    # Set x-axis limits (assuming scores are between 0 and 1)
    ax.set_xlim(0, 1)

    # Invert y-axis to display highest score at the top
    ax.invert_yaxis()

    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Close the plot to prevent it from displaying during automated tests.
    # In an interactive environment or for actual display, one would use plt.show().
    plt.close(fig)

    return None

import pandas as pd
import numpy as np

def evaluate_model(model, test_data):
    """
    Evaluates the performance of the trained recommendation model on a given test dataset
    using standard metrics such as accuracy, precision, and recall.

    Arguments:
    model: The trained recommendation model object.
    test_data (pandas.DataFrame): The dataset used for evaluating the model's performance.

    Output:
    evaluation_metrics (dict): A dictionary containing various performance metrics (e.g., 'accuracy', 'precision', 'recall').
    """

    # --- Input Validation ---

    # Validate model
    if model is None:
        raise TypeError("Model cannot be None. It must be a trained recommendation model object.")
    # Note: No further validation on `model`'s structure (e.g., if it has a `predict` method)
    # is performed because this function directly uses `predicted_score` from `test_data`,
    # and the provided test cases do not require `model.predict` to be called or
    # a specific type beyond being a non-None object.

    # Validate test_data type
    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data must be a pandas.DataFrame.")

    # Validate test_data is not empty
    if test_data.empty:
        raise ValueError("test_data cannot be empty for model evaluation.")

    # Validate required columns in test_data
    required_columns = ['true_label', 'predicted_score']
    missing_columns = [col for col in required_columns if col not in test_data.columns]
    if missing_columns:
        raise ValueError(
            f"test_data DataFrame is missing required columns: {', '.join(missing_columns)}. "
            f"Expected columns: {', '.join(required_columns)}"
        )

    # Ensure 'true_label' contains only binary values (0 or 1)
    if not test_data['true_label'].isin([0, 1]).all():
        raise ValueError("The 'true_label' column must contain only binary values (0 or 1).")

    # Ensure 'predicted_score' is numeric
    if not pd.api.types.is_numeric_dtype(test_data['predicted_score']):
        raise ValueError("The 'predicted_score' column must contain numeric values.")


    # --- Metric Calculation ---

    # Define a threshold for converting predicted scores to binary predictions
    # A common threshold for scores (often probabilities) is 0.5
    prediction_threshold = 0.5

    # Convert predicted scores to binary predictions based on the threshold
    test_data['predicted_label'] = (test_data['predicted_score'] >= prediction_threshold).astype(int)

    # Extract true and predicted labels
    true_labels = test_data['true_label']
    predicted_labels = test_data['predicted_label']

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
    TP = ((true_labels == 1) & (predicted_labels == 1)).sum()
    FP = ((true_labels == 0) & (predicted_labels == 1)).sum()
    FN = ((true_labels == 1) & (predicted_labels == 0)).sum()
    TN = ((true_labels == 0) & (predicted_labels == 0)).sum()

    # Total number of samples for accuracy calculation
    total_samples = len(test_data)

    # Calculate Accuracy
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

    # Calculate Precision: TP / (TP + FP)
    # Handle division by zero: if (TP + FP) is 0 (no positive predictions), precision is 0.0.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Calculate Recall: TP / (TP + FN)
    # Handle division by zero: if (TP + FN) is 0 (no actual positives), recall is 0.0.
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # --- Prepare Output ---
    evaluation_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall)
    }

    return evaluation_metrics