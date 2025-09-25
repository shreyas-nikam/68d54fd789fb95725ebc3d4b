
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will provide an interactive environment for users to explore the concept of fairness in course recommendation systems. It will allow users to generate synthetic datasets with adjustable bias and visualize various aspects of the data to understand how biases can manifest. The application will serve as a foundational tool for learning about fairness-aware AI in educational contexts.

### Learning Goals
The application aims to help users:
*   Understand how recommendation systems work and how they can potentially perpetuate existing biases.
*   Learn about key fairness metrics, including Demographic Parity, Equal Opportunity, Equalized Odds, and Predictive Parity, and understand their implications.
*   Explore techniques to incorporate fairness constraints into a basic recommendation algorithm (though the current notebook focuses on data generation and exploration, this goal implies future extensions or a conceptual understanding from the provided text).
*   Understand the trade-offs between fairness, accuracy, and diversity in recommendation systems.

## 2. User Interface Requirements

### 2.1 Layout and Navigation Structure
The application will utilize a two-panel layout common in Streamlit:
*   **Sidebar:** Will contain all input widgets and controls for data generation parameters.
*   **Main Content Area:** Will display narrative text, generated dataframes, and interactive visualizations. Sections will be logically ordered, mirroring the Jupyter notebook flow.

### 2.2 Input Widgets and Controls
The application will provide the following interactive widgets in the sidebar for users to control the synthetic data generation:

*   **Number of Students (`num_students`):**
    *   **Type:** `st.slider` (Integer)
    *   **Range:** 100 to 5000 (default: 1000)
    *   **Tooltip:** "Adjust the total number of synthetic student profiles to generate."
*   **Number of Courses (`num_courses`):**
    *   **Type:** `st.slider` (Integer)
    *   **Range:** 10 to 500 (default: 100)
    *   **Tooltip:** "Adjust the total number of synthetic courses to generate."
*   **Bias Strength (`bias_strength`):**
    *   **Type:** `st.slider` (Float)
    *   **Range:** 0.0 to 1.0 (default: 0.2)
    *   **Tooltip:** "Control the strength of bias introduced in student-course interactions. Higher values amplify the correlation between student major and course subject area."

### 2.3 Visualization Components
The main content area will display the following:

*   **Generated DataFrames:** Interactive tables for the `students_df`, `courses_df`, and `interactions_df` (using `st.dataframe`) to allow users to inspect the raw data.
*   **Distribution Plots:**
    *   **Student GPA Distribution:** A histogram or kernel density estimate (KDE) plot (using `seaborn.histplot` or `seaborn.kdeplot`) to visualize the spread of GPAs among students.
    *   **Major Distribution:** A bar chart (using `seaborn.countplot`) to show the count of students in each major.
    *   **Course Difficulty Distribution:** A bar chart (using `seaborn.countplot`) to show the count of courses by difficulty level.
    *   **Course Subject Area Distribution:** A bar chart (using `seaborn.countplot`) to show the count of courses by subject area.
*   **Interaction Pattern Visualization:**
    *   **Average Interaction Score by Major and Subject Area:** A heatmap (using `seaborn.heatmap`) displaying the average `interaction_score` for each combination of student `major` and course `subject_area`. This will be crucial for observing the effect of `bias_strength`.

All plots will be rendered using `matplotlib.pyplot` and displayed in Streamlit using `st.pyplot`. Each plot will have a clear title, labeled axes, and legends where necessary, adhering to color-blind-friendly palettes and font sizes $\geq 12 \text{ pt}$.

### 2.4 Interactive Elements and Feedback Mechanisms
*   All input widget changes (`num_students`, `num_courses`, `bias_strength`) will automatically re-run the data generation and update all displayed dataframes and visualizations in real-time.
*   Inline help text (tooltips) will be provided for all input controls as specified in Section 2.2.
*   The application will provide clear textual explanations accompanying each section, including descriptions of the data generation process and interpretations of the visualizations.

## 3. Additional Requirements

### 3.1 Annotation and Tooltip Specifications
*   **Input Widgets:** Each slider/input box will have a descriptive tooltip explaining its purpose and impact on the synthetic data.
*   **Visualizations:** Tooltips on interactive charts (if supported by libraries like Altair or Plotly, or by custom event handlers in Streamlit) could show specific data points or aggregated values on hover. For `matplotlib`/`seaborn`, clear titles and axis labels will serve as primary annotations.

### 3.2 Save the states of the fields properly so that changes are not lost
The application will utilize `st.session_state` to persist the values of the input widgets (`num_students`, `num_courses`, `bias_strength`). This ensures that if the user navigates away or refreshes the page, their selected parameters are retained, and the application's state is preserved.

## 4. Notebook Content and Code Requirements

This section outlines how the content from the Jupyter notebook will be integrated into the Streamlit application.

### 4.1 Extracted Code Stubs and Usage

The following code stubs will be used in the Streamlit application:

**1. Library Imports:**
These imports will be placed at the top of the Streamlit script.

```python
import pandas as pd
import numpy as np
# For future ML models:
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # Streamlit specific import
```

**2. Synthetic Data Generation Function:**
The `generate_synthetic_data` function will be defined early in the script, likely with `@st.cache_data` decorator if performance is a concern (re-run only if inputs change).

```python
# @st.cache_data # Consider caching if data generation is computationally intensive
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
```

**3. Application Logic (Main Script):**
The Streamlit application will structure the calls to the `generate_synthetic_data` function and display the outputs based on user inputs.

```python
# Streamlit app title
st.title("Fairness-Aware Course Recommendation System")

# Sidebar for controls
st.sidebar.header("Data Generation Parameters")

# Initialize session state for parameters if not already set
if 'num_students' not in st.session_state:
    st.session_state.num_students = 1000
if 'num_courses' not in st.session_state:
    st.session_state.num_courses = 100
if 'bias_strength' not in st.session_state:
    st.session_state.bias_strength = 0.2

# Input widgets using st.session_state
st.session_state.num_students = st.sidebar.slider(
    "Number of Students", 100, 5000, value=st.session_state.num_students,
    help="Adjust the total number of synthetic student profiles to generate."
)
st.session_state.num_courses = st.sidebar.slider(
    "Number of Courses", 10, 500, value=st.session_state.num_courses,
    help="Adjust the total number of synthetic courses to generate."
)
st.session_state.bias_strength = st.sidebar.slider(
    "Bias Strength", 0.0, 1.0, value=st.session_state.bias_strength, step=0.05,
    help="Control the strength of bias introduced in student-course interactions. Higher values amplify the correlation between student major and course subject area."
)

# Generate data based on inputs
students_df, courses_df, interactions_df = generate_synthetic_data(
    st.session_state.num_students, st.session_state.num_courses, st.session_state.bias_strength
)

# Display DataFrames (replacing print statements from notebook)
st.subheader("Generated Student Data (first 5 rows):")
st.dataframe(students_df.head())
st.subheader("Generated Course Data (first 5 rows):")
st.dataframe(courses_df.head())
st.subheader("Generated Interaction Data (first 5 rows):")
st.dataframe(interactions_df.head())

# Placeholder for visualization code based on "Exploring the Data"
st.header("Exploring the Data")

# GPA Distribution
st.subheader("Distribution of Student GPAs")
fig_gpa, ax_gpa = plt.subplots()
sns.histplot(students_df['gpa'], bins=10, kde=True, ax=ax_gpa)
ax_gpa.set_title("Distribution of Student GPAs")
ax_gpa.set_xlabel("GPA")
ax_gpa.set_ylabel("Number of Students")
st.pyplot(fig_gpa)

# Major Distribution
st.subheader("Distribution of Student Majors")
fig_major, ax_major = plt.subplots()
sns.countplot(y='major', data=students_df, ax=ax_major, palette='viridis')
ax_major.set_title("Distribution of Student Majors")
ax_major.set_xlabel("Number of Students")
ax_major.set_ylabel("Major")
st.pyplot(fig_major)

# Course Subject Area Distribution
st.subheader("Distribution of Course Subject Areas")
fig_subject, ax_subject = plt.subplots()
sns.countplot(y='subject_area', data=courses_df, ax=ax_subject, palette='magma')
ax_subject.set_title("Distribution of Course Subject Areas")
ax_subject.set_xlabel("Number of Courses")
ax_subject.set_ylabel("Subject Area")
st.pyplot(fig_subject)

# Interaction Pattern Heatmap (Major vs Subject Area)
st.subheader("Average Interaction Score by Major and Course Subject Area")
if not interactions_df.empty:
    merged_df = interactions_df.merge(students_df[['student_id', 'major']], on='student_id') \
                               .merge(courses_df[['course_id', 'subject_area']], on='course_id')
    
    pivot_table = merged_df.pivot_table(index='major', columns='subject_area', values='interaction_score', aggfunc='mean')
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_heatmap)
    ax_heatmap.set_title("Average Interaction Score by Student Major and Course Subject Area")
    ax_heatmap.set_xlabel("Course Subject Area")
    ax_heatmap.set_ylabel("Student Major")
    st.pyplot(fig_heatmap)
else:
    st.warning("No interactions generated. Adjust parameters to generate interactions.")

# Optional: Display library versions (can be put in an expander)
with st.expander("Library Versions"):
    st.write(f"Pandas Version: {pd.__version__}")
    st.write(f"Numpy Version: {np.__version__}")
    # st.write(f"Scikit-learn Version: {sklearn.__version__}") # Uncomment if sklearn is used
    st.write(f"Matplotlib Version: {matplotlib.__version__}")
    st.write(f"Seaborn Version: {seaborn.__version__}")
    st.write(f"Streamlit Version: {st.__version__}")
```

### 4.2 Markdown Content
All markdown cells from the Jupyter notebook will be converted to Streamlit markdown using `st.markdown` or `st.header`/`st.subheader`/`st.write`.

**Initial Markdown:**
```markdown
# Fairness-Aware Course Recommendation System

Recommendation systems are ubiquitous in our daily lives, influencing everything from what movies we watch to what products we buy. In educational settings, they hold immense potential to guide students towards courses that align with their interests, career goals, and academic strengths. However, these systems, like any data-driven technology, are not immune to biases present in their training data. If left unaddressed, these biases can perpetuate or even amplify existing inequalities, leading to unfair or discriminatory recommendations for certain student groups.

This notebook aims to delve into the critical aspect of fairness in course recommendation systems. We will explore how recommendation systems can inadvertently perpetuate existing biases and, more importantly, learn techniques to mitigate these biases and build more equitable systems. We will cover the following learning goals:

*   Understand how recommendation systems work and how they can potentially perpetuate existing biases.
*   Learn about key fairness metrics, including Demographic Parity, Equal Opportunity, Equalized Odds, and Predictive Parity, and understand their implications.
*   Explore techniques to incorporate fairness constraints into a basic recommendation algorithm.
*   Understand the trade-offs between fairness, accuracy, and diversity in recommendation systems.

Our exploration will be guided by principles of fairness in AI, referencing contemporary research such as FairAIE [32](#references).
```

**Explanation of Libraries Markdown:**
```markdown
### Explanation of Libraries

This notebook leverages several powerful Python libraries to achieve its objectives:

*   **pandas**: A fundamental library for data manipulation and analysis. It provides DataFrames, which are efficient and flexible data structures for working with tabular data.
*   **numpy**: Essential for numerical computations in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
*   **scikit-learn**: A comprehensive machine learning library that provides a wide range of algorithms for classification, regression, clustering, and dimensionality reduction. We will specifically use `train_test_split` for data partitioning, `LogisticRegression` for content-based recommendation, and `TruncatedSVD` for collaborative filtering.
*   **matplotlib.pyplot**: A plotting library that provides a MATLAB-like interface for creating static, interactive, and animated visualizations in Python.
*   **seaborn**: A data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
*   **ipywidgets**: A library that enables the creation of interactive controls (e.g., sliders, dropdowns) directly within Jupyter notebooks, allowing for dynamic exploration of parameters and results. (Note: In Streamlit, `ipywidgets` will be replaced by Streamlit's native widgets).
```

**Generating Synthetic Data Markdown:**
```markdown
## Generating Synthetic Data

To effectively explore fairness in recommendation systems without relying on real-world datasets that might be proprietary or contain sensitive information, we will generate a synthetic dataset. This synthetic data will simulate student profiles, course information, and student-course interactions, allowing us to control and introduce biases for experimental purposes.

The synthetic dataset will consist of three main components:

1.  **Student Profiles**: Contains information about individual students, such as their academic performance and chosen field of study.
2.  **Course Information**: Details about various courses offered, including their difficulty and subject area.
3.  **Student-Course Interactions**: Records of how students have interacted with courses (e.g., enrollment, ratings), which will be used to train our recommendation model.

The generation process allows us to control key parameters such as the number of students, the number of courses, and a `bias_strength` factor. This `bias_strength` is crucial for our fairness analysis, as it will simulate real-world scenarios where certain student demographics (e.g., students from particular majors) might have a higher propensity to interact with specific course subject areas. For instance, students majoring in "Computer Science" might be inherently more likely to enroll in "STEM" courses, and the bias strength will amplify this correlation.
```

**Explanation of Data Generation Markdown:**
```markdown
### Explanation of Data Generation

The `generate_synthetic_data` function creates three interconnected Pandas DataFrames:

1.  **`students_df`**: This DataFrame contains simulated student profiles with the following columns:
    *   `student_id` (int): A unique identifier for each student. This is our primary key, and we assert its uniqueness.
    *   `gpa` (float): The Grade Point Average of the student, a numeric value between 2.0 and 4.0.
    *   `major` (category): The student's declared major, chosen from a predefined list (e.g., 'Computer Science', 'Literature'). This will serve as our sensitive attribute for fairness analysis.

2.  **`courses_df`**: This DataFrame holds information about the courses available, with these columns:
    *   `course_id` (int): A unique identifier for each course. This is also a primary key and is asserted to be unique.
    *   `difficulty` (category): The perceived difficulty of the course (e.g., 'Easy', 'Medium', 'Hard').
    *   `subject_area` (category): The broad subject category of the course (e.g., 'STEM', 'Humanities', 'Business'). This is used in conjunction with `major` to introduce bias in interactions.

3.  **`interactions_df`**: This DataFrame records the interactions (e.g., enrollment interest, historical engagement) between students and courses. It includes:
    *   `student_id` (int): The identifier of the student involved in the interaction. This is a foreign key referencing `students_df`.
    *   `course_id` (int): The identifier of the course involved in the interaction. This is a foreign key referencing `courses_df`.
    *   `interaction_score` (float): A normalized score (between 0.0 and 1.0) representing the strength or likelihood of a student's interaction with a course. This score is influenced by the `bias_strength` parameter, where students are more likely to have higher interaction scores with courses in subject areas aligned with their major.

#### Dataset Information and Handling/Validation Requirements:

*   **Expected Column Names and Data Types**: The columns and their types (e.g., `int` for IDs, `float` for GPA/scores, `category` for nominal data) are strictly defined to ensure data integrity and compatibility with downstream machine learning algorithms.
*   **Primary-Key Uniqueness**: Both `student_id` in `students_df` and `course_id` in `courses_df` are designed to be unique, guaranteeing that each student and course is uniquely identifiable. This is crucial for accurate data merging and relationship establishment.
*   **No Missing Values**: The data generation process is designed to prevent missing values in critical fields. Assertions or checks can be added to confirm this, though the synthetic generation itself ensures completeness.
*   **Summary Statistics**: We will log summary statistics for numeric columns (e.g., GPA, interaction score) to understand their distribution, central tendency, and spread.

Now, let's examine the first few rows and some basic statistics of our generated DataFrames to confirm their structure and content.
```

**Exploring the Data Markdown:**
```markdown
## Exploring the Data

Understanding the characteristics and distributions within our dataset is a crucial first step before building any recommendation model. Data exploration helps us identify potential biases, understand the relationships between different attributes, and gain insights that can inform our modeling approach.

In this section, we will visualize various aspects of our synthetic data to:

*   Examine the distribution of student GPAs.
*   Understand the distribution of courses across different subject areas.
*   Visualize the student-course interaction patterns, especially to observe the effect of the `bias_strength` we introduced.

These visualizations will provide a foundational understanding of our data, highlight any inherent imbalances, and set the stage for our fairness evaluation.
```

**Fairness in AI in Educational Settings and Notations:**
All content from "4 Fairness in AI in Educational Settings" to the end of "4.3.3 Fairness Bonded Utility" will be included using `st.markdown`, ensuring strict LaTeX formatting.

*   **Section 4.1 Notations:**
    *   $D$: binary classification dataset for a class attribute $Y = \{+, -\}$.
    *   $A$: binary sensitive attribute, where $A \in \{a, \bar{a}\}$. $a$ is the protected group, $\bar{a}$ is the non-protected group.
    *   $\hat{Y}$: predicted outcome of the classification, $\hat{Y} = \{+, -\}$.
    *   $a^+$ and $a^-$: instances where the protected group are classified as positive and negative, respectively.
    *   $\bar{a}^+$ and $\bar{a}^-$: instances where the non-protected group are classified as positive and negative, respectively.
    *   Metrics: True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP).
    *   $f(\cdot)$: predictive model function.
    *   $dY(\cdot)$: distance metric in output space.
    *   $dX(\cdot)$: distance metric in input space.
    *   $L$: Lipschitz constant.
    *   $y_i$: probabilistic classification output for instance $x_i$.
    *   $x_j$: top $k$-nearest neighbors for $x_i$.

    **Table 1: Notations** (will be rendered as `st.dataframe` or manually formatted markdown table)
    | Notation | Definition |
    |---|---|
    | $Y$ | Class attribute with binary values $\{+,-\}$, representing the positive and negative outcomes in model. |
    | $A$ | Binary sensitive attribute with values $\{a, \bar{a}\}$, indicating membership in a protected or non-protected group. |
    | $a$ | Protected group, the group that is safe against discrimination by fairness constraints. |
    | $\bar{a}$ | Non-protected group, the group that is not main focus of fairness measures but is important for analysis. |
    | $\hat{Y}$ | Predicted outcome by the model, aims to approximate the true class attribute $Y$. |
    | TP | True Positive, where the model correctly predicts a positive outcome. |
    | FN | False Negative, where the model incorrectly predicts a negative outcome for a positive case. |
    | TN | True Negative, where the model correctly predicts a negative outcome. |
    | FP | False Positive, where the model incorrectly predicts a positive outcome for a negative case. |
    | $f(\cdot)$ | Predictive model function, mapping input features to the predicted outcome. |
    | $dY(\cdot)$ | Distance metric in output space, measuring similarity between predicted and actual outcomes. |
    | $dX(\cdot)$ | Distance metric in input space, quantifying the difference between predicted and actual outcomes. |
    | $L$ | Lipschitz constant, that rescales input distance between input variables. |
    | $y_i$ | Probabilistic classification output for instance $x_i$. |
    | $x_j$ | Top $k$-nearest neighbors for $x_i$, used in fairness analysis to evaluate local decision boundaries. |

    The notation $x_j$ refers to the top $k$-nearest neighbors of $x_i$, important for analyzing local fairness at decision boundaries.

*   **Section 4.2 Fairness Notions:**
    *   **Individual Fairness:** Principle that students with similar academic abilities and backgrounds should receive similar treatment.
    *   **Group Fairness:** Principle that groups of students should be treated fairly, regardless of their individual characteristics.

*   **Section 4.3 Fairness Evaluating Resources:**
    *   **4.3.1 Individual Fairness:**
        *   **Distance-Based Individual Fairness:** Adheres to the Lipschitz condition.
        $$ dy(f(x_i), f(x_j)) \leq L \cdot dx(x_i, x_j) $$
        Where $f(\cdot)$ is the predictive model, $dY(\cdot)$ and $dX(\cdot)$ are distance metrics, and $L$ is the Lipschitz constant.
        The average distance of the output between each individual and its $k$-nearest neighbors:
        $$ 1 - \frac{1}{n \cdot k} \sum_{i=1}^{n} \left| y_i - \sum_{j=1}^{k} \overline{Y_j} \right| $$
        *   **Ranking-Based Individual Fairness:** Uses ranking similarity metrics like Normalized Discounted Cumulative Gain (NDCG@k) and Expected Reciprocal Rank (ERR@k).

    *   **4.3.2 Group Fairness:**
        *   **Statistical Parity:** Ensures equal outcome distribution across demographic groups within tolerance $\epsilon$.
        $$ P(\hat{Y}|A = a) - P(\hat{Y}|A = \bar{a}) \leq \epsilon. $$
        Calculated as:
        $$ \text{Statistical Parity} = P(\hat{Y} = +|A = a) - P(\hat{Y} = +|A = \bar{a}). $$
        *   **Equal Opportunity:** Focuses on fairness regarding positive outcomes when the true outcome is positive.
        $$ P(\hat{Y} = +|A = a, Y = +) = P(\hat{Y} = +|A = \bar{a}, Y = +). $$
        This means equal True Positive Rates (TPR):
        $$ \text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}} $$
        A classifier with equal False Negative Rates (FNR) will also have equal TPRs:
        $$ \text{FNR} = \frac{\text{FN}}{\text{TP}+\text{FN}} $$
        Quantified by:
        $$ \text{Equal Opportunity} = \left| P(\hat{Y} = -|Y = +, A = a) - P(\hat{Y} = -|Y = +, A = \bar{a}) \right|. $$
        *   **Equalized Odds:** Predicted outcome $\hat{Y}$ and sensitive attribute $A$ are independent given true outcome $Y$.
        $$ P(\hat{Y} = +|A = a, Y = y) = P(\hat{Y} = +|A = \bar{a}, Y = y), \quad \text{where } y \in \{+,-\}. $$
        Formulated as:
        $$ \text{Equalized Odds} = \sum_{y \in \{+,-\}} \left| P(\hat{Y} = +|A = a, Y = y) - P(\hat{Y} = +|A = \bar{a}, Y = y) \right|. $$
        *   **Predictive Parity:** Both protected and non-protected groups have an equal Positive Predictive Value (PPV).
        $$ \text{PPV} = \frac{\text{TP}}{\text{TP}+\text{FP}} $$
        Formally:
        $$ P(Y = +|\hat{Y} = +, A = a) = P(Y = +|\hat{Y} = +, A = \bar{a}) $$
        Reformulated as:
        $$ \text{Predictive Parity} = \left| P(Y = +|\hat{Y} = +, A = a) - P(Y = +|\hat{Y} = +, A = \bar{a}) \right|. $$
        *   **Predictive Equality:** Also False Positive Rate (FPR) balance.
        $$ P(\hat{Y} = +|Y = -, A = a) = P(\hat{Y} = +|Y = -, A = \bar{a}) $$
        Measured by the difference:
        $$ \text{Predictive Equality} = \left| P(\hat{Y} = +|Y = -, A = a) - P(\hat{Y} = +|Y = -, A = \bar{a}) \right|. $$

    *   **4.3.3 Fairness Bonded Utility (FBU):** Addresses the trade-off between model performance and fairness, using a two-dimensional coordinate system and categorizing techniques into five effectiveness levels (Level 1: Jointly Advantageous, Level 2: Impressive, Level 3: Reversed, Level 4: Deficient, Level 5: Jointly Disadvantageous). The visualization will include Figure 2 (Graph a and b) to illustrate these concepts.

*   **Section 4.4 Approaches to Improve Fairness:** Includes Pre-processing (Re-weighing, Resampling: Oversampling, Undersampling), In-processing (Regularization Methods, Constraint-based Optimization, Adversarial Debiasing), and Post-processing (Threshold Adjustment, Outcome Perturbation). Figure 3 will be used to illustrate the overall architecture.

*   **Section 5 Ethical Considerations and Frameworks in AI for Education:** Discussion on ethical dilemmas, guidelines, policies, and frameworks (Table 2 will be formatted as `st.dataframe` or markdown table).

*   **Section 6 Tools and Datasets:** Lists various tools (AI Fairness 360, Fairlearn, TensorFlow Fairness Indicators, What-If Tools, FairTest, FairML, Aequitas) and datasets (TIMSS, PISA, NELS:88, EdNet, KDDCUP 2015, OULAD, etc.) used in fairness research, along with Table 3.

*   **Section 7 Challenges and Future Directions:** Discussion on Personalization Versus Data Privacy, Data-Related Challenges, The Definition and Measurement of Fairness, Trade-offs Between Fairness and Accuracy, Digital-divide, and Teacher and Student Adoption.

*   **References:** A detailed "References" section will be included at the end of the application, listing all cited works from the notebook, although the license itself will be excluded.
