
# Technical Specification for Jupyter Notebook: AI Fairness in Medical Curriculum Builder

## 1. Notebook Overview

This Jupyter Notebook provides a practical framework for medical educators to understand, identify, and mitigate algorithmic biases in AI-integrated curricula. It leverages synthetic medical student data to demonstrate how biases related to gender and socio-economic status can manifest in predictive models and explores techniques to promote equitable learning experiences.

**Learning Goals:**

*   Understand the importance of fairness in medical AI.
*   Learn how algorithmic biases can impact educational outcomes and experiences of medical students.
*   Explore techniques to mitigate biases using pre-processing strategies.
*   Learn how to integrate ethical considerations into medical curriculum design through practical bias analysis.
*   Gain familiarity with fairness metrics and their interpretation.

## 2. Code Requirements

### List of Expected Libraries

The following open-source Python libraries (from PyPI) are expected to be used:

*   `numpy`: For numerical operations, especially in data generation.
*   `pandas`: For data manipulation and analysis, handling DataFrames.
*   `sklearn.model_selection.train_test_split`: For splitting data into training and testing sets.
*   `sklearn.preprocessing.LabelEncoder`, `sklearn.preprocessing.StandardScaler`: For encoding categorical features and scaling numerical features.
*   `sklearn.linear_model.LogisticRegression`: For training a baseline classification model.
*   `sklearn.metrics.classification_report`, `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`: For evaluating model performance.
*   `matplotlib.pyplot`: For static plotting and visualization.
*   `seaborn`: For enhanced statistical data visualization.
*   `ipywidgets`: For creating interactive user controls (sliders, dropdowns) to enable dynamic analysis.
*   `aif360.datasets.StandardDataset`: For converting pandas DataFrames into an AIF360-compatible format.
*   `aif360.metrics.BinaryLabelDatasetMetric`: For calculating fairness metrics on datasets.
*   `aif360.metrics.ClassificationMetric`: For calculating fairness metrics on classification models.
*   `aif360.algorithms.preprocessing.Reweighing`: For applying a pre-processing bias mitigation technique.

### List of Algorithms or Functions to be Implemented

1.  **`generate_synthetic_medical_data(num_samples: int) -> pandas.DataFrame`**: Generates a synthetic dataset of medical student records with specified features including sensitive attributes and a target variable.
2.  **`validate_and_summarize_data(df: pandas.DataFrame) -> None`**: Performs data validation (checks column names, data types, missing values, primary key uniqueness) and displays summary statistics for numeric columns.
3.  **`preprocess_data(df: pandas.DataFrame, sensitive_attribute_names: list, target_label_name: str, favorable_label: int, protected_attribute_map: dict) -> tuple`**: Encodes categorical features, scales numerical features, defines privileged/unprivileged groups, and splits data into training/testing sets, returning features, target, and AIF360 `StandardDataset` objects.
4.  **`train_logistic_regression_model(X_train: pandas.DataFrame, y_train: pandas.Series, sample_weights: numpy.ndarray = None) -> sklearn.linear_model.LogisticRegression`**: Trains a Logistic Regression model.
5.  **`evaluate_model_performance(model: sklearn.linear_model.LogisticRegression, X_test: pandas.DataFrame, y_test: pandas.Series) -> dict`**: Evaluates a model's performance using standard metrics (accuracy, precision, recall, F1-score).
6.  **`calculate_fairness_metrics(model: sklearn.linear_model.LogisticRegression, dataset_test: aif360.datasets.StandardDataset, privileged_groups: list, unprivileged_groups: list) -> dict`**: Calculates various fairness metrics (Statistical Parity Difference, Equal Opportunity Difference, Average Odds Difference) for a given model and test dataset using AIF360.
7.  **`plot_feature_distribution(df: pandas.DataFrame, feature: str, hue_feature: str) -> None`**: Generates a bar plot or histogram to show the distribution of a feature, potentially broken down by another feature.
8.  **`plot_outcome_distribution_by_sensitive_attribute(df: pandas.DataFrame, sensitive_attribute: str, target_attribute: str) -> None`**: Generates a stacked bar chart to show the distribution of the target outcome across different groups of a sensitive attribute.
9.  **`plot_fairness_metrics_comparison(baseline_metrics: dict, mitigated_metrics: dict, metric_name: str, sensitive_attribute: str) -> None`**: Creates a bar plot to compare specific fairness metrics between baseline and mitigated models.
10. **`plot_fairness_accuracy_tradeoff(model, dataset_test, privileged_groups, unprivileged_groups) -> None`**: Generates an interactive plot showing the trade-off between a fairness metric and accuracy across a range of prediction thresholds.

### Visualization Requirements

The notebook should generate the following types of visualizations:

1.  **Distribution Plots**: Bar charts and histograms (e.g., `seaborn.countplot`, `seaborn.histplot`) for understanding feature distributions, especially sensitive attributes and the target variable.
2.  **Relationship Plots**: Pair plots (e.g., `seaborn.pairplot`) or scatter plots to examine correlations between key features and potential relationships with sensitive attributes.
3.  **Aggregated Comparison Plots**: Stacked bar charts (e.g., `pandas.DataFrame.plot(kind='bar', stacked=True)`) or heatmaps (`seaborn.heatmap`) to compare outcomes and feature aggregates across different categorical groups (e.g., pass/fail rates by gender or SES).
4.  **Fairness Metric Comparison Plots**: Bar plots (`seaborn.barplot` or `matplotlib.pyplot.bar`) to visually compare fairness metric values (e.g., Statistical Parity Difference, Equal Opportunity Difference) between baseline and bias-mitigated models.
5.  **Fairness-Accuracy Trade-off Plot**: A line plot (interactive using `ipywidgets` and `matplotlib.pyplot` or `plotly.express` for a more advanced version) showing how fairness metrics and accuracy change with varying model prediction thresholds or mitigation parameter strengths. This serves as the "trend plot" by showing a trend of these metrics over a continuous parameter.
6.  **Confusion Matrix Visualization**: A heatmap (`seaborn.heatmap`) to visualize the confusion matrix for the baseline and mitigated models, possibly by group, to aid in understanding performance disparities.

All plots should adhere to the following style and usability guidelines:
*   Adopt a color-blind-friendly palette (e.g., `seaborn.color_palette("viridis")`).
*   Ensure font size $\geq$ 12 pt for readability.
*   Supply clear titles, labeled axes, and legends.
*   Enable interactivity where the environment supports it (`ipywidgets` for control, `matplotlib`'s interactive backend, or `plotly`).
*   Offer a static fallback (saved PNG) when interactive libraries are unavailable or for documentation.

## 3. Notebook Sections (in Detail)

---

### Section 1: Introduction to AI Fairness in Medical Education

*   **Markdown Cell:**
    Explains the purpose of the notebook: to explore fairness, bias, and ethics in AI-integrated medical curricula. It highlights the potential for AI to perpetuate or amplify existing societal biases, particularly concerning protected groups defined by gender and socio-economic status, drawing from the overview and references [32], [33]. It briefly introduces the concept of fairness metrics and mitigation strategies.

*   **Code Cell:**
    *(No code in this section)*

*   **Code Cell:**
    *(No execution in this section)*

*   **Markdown Cell:**
    *(No explanation for execution)*

---

### Section 2: Setup and Library Imports

*   **Markdown Cell:**
    Explains that this section will import all necessary Python libraries. It emphasizes that only open-source libraries from PyPI are used, ensuring compatibility and ease of execution.

*   **Code Cell:**
    ```python
    # Code Cell: Library Imports
    # This cell imports all required libraries for data handling, machine learning,
    # visualization, and fairness analysis using AIF360.
    # It ensures the notebook has access to all necessary functionalities.
    # Expected imports: numpy, pandas, train_test_split, LabelEncoder, StandardScaler,
    # LogisticRegression, classification_report, accuracy_score, confusion_matrix,
    # matplotlib.pyplot, seaborn, ipywidgets, StandardDataset, BinaryLabelDatasetMetric,
    # ClassificationMetric, Reweighing.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Library Imports
    # This cell executes the imports. There is no specific function call here,
    # just the import statements themselves.
    ```

*   **Markdown Cell:**
    Confirms that all libraries have been successfully loaded, preparing the environment for data generation and analysis.

---

### Section 3: Synthetic Medical Student Dataset Generation

*   **Markdown Cell:**
    This section focuses on creating a synthetic dataset that mimics medical student demographics and performance. The dataset will include `Student_ID`, `Gender`, `SES_Level`, `Admission_Exam_Score`, `Interview_Score`, `Clinical_Rotation_Grade`, `Research_Experience`, `Year_of_Admission`, and `Medical_School_Performance` (the binary target variable). Biases related to `Gender` and `SES_Level` will be intentionally introduced during generation to simulate real-world disparities, making it suitable for fairness analysis.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Data Generation
    # This cell defines a function to generate synthetic medical student data.
    # Function: generate_synthetic_medical_data(num_samples: int) -> pandas.DataFrame
    #   - num_samples: The number of synthetic student records to generate.
    #   - Returns: A pandas DataFrame containing the synthetic data with specified columns and data types.
    #   - Introduces synthetic biases: For example, 'Female' students from 'Low' SES_Level might have slightly lower
    #     Admission_Exam_Score or Clinical_Rotation_Grade on average, influencing the target 'Medical_School_Performance'.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Data Generation
    # This cell calls the `generate_synthetic_medical_data` function to create a dataset
    # of 1000 synthetic medical student records and stores it in a DataFrame named `df_medical_students`.
    ```

*   **Markdown Cell:**
    Explains the structure of the newly generated `df_medical_students` DataFrame, including a preview of its head, shape, and column types. It verifies that the synthetic data adheres to the expected schema.

---

### Section 4: Data Exploration and Validation

*   **Markdown Cell:**
    This section performs crucial data validation and initial exploration. It verifies expected column names, data types, and checks for primary-key uniqueness (`Student_ID`). It asserts no missing values in critical fields and logs summary statistics for all numeric columns. Visualizations are generated to understand the distribution of key features, especially `Gender`, `SES_Level`, and `Medical_School_Performance`.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Data Validation and Summary
    # This function validates the DataFrame structure and provides summary statistics.
    # Function: validate_and_summarize_data(df: pandas.DataFrame) -> None
    #   - df: The input DataFrame to validate and summarize.
    #   - Actions: Confirms expected column names and types, checks 'Student_ID' for uniqueness,
    #     asserts no missing values in key columns ('Gender', 'SES_Level', 'Medical_School_Performance'),
    #     and prints descriptive statistics for numeric fields.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Data Validation and Summary
    # This cell executes the `validate_and_summarize_data` function on `df_medical_students`.
    # It also generates distribution plots for 'Gender', 'SES_Level', and 'Medical_School_Performance'
    # using `seaborn.countplot` with a color-blind friendly palette and appropriate labels.
    # Expected plots: Bar plots for 'Gender', 'SES_Level', and 'Medical_School_Performance'.
    plot_feature_distribution(df_medical_students, 'Gender', None)
    plot_feature_distribution(df_medical_students, 'SES_Level', None)
    plot_feature_distribution(df_medical_students, 'Medical_School_Performance', None)
    plot_outcome_distribution_by_sensitive_attribute(df_medical_students, 'Gender', 'Medical_School_Performance')
    plot_outcome_distribution_by_sensitive_attribute(df_medical_students, 'SES_Level', 'Medical_School_Performance')
    ```

*   **Markdown Cell:**
    Analyzes the output from the validation and summary, noting any anomalies or interesting distributions. Interprets the initial visualizations, observing any apparent imbalances in the sensitive attributes or the target variable across groups. For example, it might highlight that 'Female' students or 'Low' SES_Level students have a lower representation in the 'Above_Average' performance group.

---

### Section 5: Data Preprocessing for Modeling

*   **Markdown Cell:**
    Prepares the dataset for machine learning. This involves encoding categorical features (`Gender`, `SES_Level`) into numerical format and scaling numerical features. The data is then split into training and testing sets. Importantly, the dataset is converted into an AIF360 `StandardDataset` format, which requires defining sensitive attributes and mapping privileged and unprivileged groups (e.g., 'Male' as privileged for 'Gender', 'High' as privileged for 'SES_Level').

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Data Preprocessing
    # This function preprocesses the data for model training and fairness analysis.
    # Function: preprocess_data(df: pandas.DataFrame, sensitive_attribute_names: list, target_label_name: str, favorable_label: int, protected_attribute_map: dict) -> tuple
    #   - df: The input DataFrame.
    #   - sensitive_attribute_names: List of column names considered sensitive (e.g., ['Gender', 'SES_Level']).
    #   - target_label_name: Name of the target variable column ('Medical_School_Performance').
    #   - favorable_label: The value representing the "favorable" outcome (e.g., 1 for 'Above_Average').
    #   - protected_attribute_map: Dictionary mapping sensitive attribute names to privileged/unprivileged values (e.g., {'Gender': {'privileged_groups': [['Female']], 'unprivileged_groups': [['Male']]}}).
    #   - Returns: X_train, y_train, X_test, y_test (pandas DataFrames/Series), and AIF360 StandardDataset objects for train/test.
    #   - Steps: Label encodes 'Gender' and 'SES_Level'. Scales numerical features. Splits data 80/20. Converts to AIF360 StandardDataset.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Data Preprocessing
    # This cell executes the `preprocess_data` function.
    # Define sensitive attribute names, target label, and favorable label.
    # Define protected_attribute_map:
    #   - For 'Gender': 'Female' as unprivileged (0), 'Male' as privileged (1).
    #   - For 'SES_Level': 'Low' and 'Medium' as unprivileged (0, 1), 'High' as privileged (2).
    # Store the preprocessed outputs in appropriate variables.
    ```

*   **Markdown Cell:**
    Summarizes the outcome of preprocessing, confirming the dimensions of the training and testing sets, and the successful conversion to AIF360 `StandardDataset` objects.

---

### Section 6: Baseline Model Training

*   **Markdown Cell:**
    Describes the training of a baseline machine learning model. A `LogisticRegression` model from `sklearn.linear_model` is chosen as a straightforward classifier. This model is trained on the preprocessed training data without any fairness interventions, serving as a point of comparison for later bias mitigation efforts.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Baseline Model Training
    # This function trains a Logistic Regression model.
    # Function: train_logistic_regression_model(X_train: pandas.DataFrame, y_train: pandas.Series, sample_weights: numpy.ndarray = None) -> sklearn.linear_model.LogisticRegression
    #   - X_train: Features for training.
    #   - y_train: Target labels for training.
    #   - sample_weights: Optional, for reweighed training (defaults to None for baseline).
    #   - Returns: A trained LogisticRegression model.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Baseline Model Training
    # This cell calls `train_logistic_regression_model` using `X_train` and `y_train` to train the baseline model.
    # The trained model is stored in `baseline_model`.
    ```

*   **Markdown Cell:**
    Confirms that the baseline Logistic Regression model has been successfully trained.

---

### Section 7: Baseline Model Evaluation and Bias Identification

*   **Markdown Cell:**
    This section evaluates the performance of the baseline model and, more critically, identifies inherent biases. Standard classification metrics (accuracy, precision, recall, F1-score) are computed. Fairness metrics are calculated using `aif360.metrics.ClassificationMetric`, specifically focusing on `Statistical Parity Difference` and `Equal Opportunity Difference`, with respect to `Gender` and `SES_Level`.

    **Statistical Parity Difference (SPD)** measures the difference in the probability of a favorable outcome between unprivileged and privileged groups.
    $$
    \text{SPD} = P(\hat{Y}=+|D=\text{unprivileged}) - P(\hat{Y}=+|D=\text{privileged})
    $$
    Where $P(\hat{Y}=+|D=\text{group})$ is the probability of a favorable prediction for a given demographic group.

    **Equal Opportunity Difference (EOD)** measures the difference in true positive rates (TPR) between unprivileged and privileged groups when the actual outcome is favorable ($Y=+$).
    $$
    \text{EOD} = P(\hat{Y}=+|D=\text{unprivileged}, Y=+) - P(\hat{Y}=+|D=\text{privileged}, Y=+)
    $$
    A value close to 0 indicates fairness.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Model Performance and Bias Evaluation
    # This function evaluates model performance and calculates fairness metrics using AIF360.
    # Function: evaluate_and_identify_bias(model, X_test, y_test, dataset_test_aif, sensitive_attribute_names, privileged_groups_map, unprivileged_groups_map) -> dict
    #   - model: The trained classification model.
    #   - X_test: Test features.
    #   - y_test: True test labels.
    #   - dataset_test_aif: AIF360 StandardDataset for the test set.
    #   - sensitive_attribute_names: List of sensitive attribute column names.
    #   - privileged_groups_map: Map of privileged groups for sensitive attributes.
    #   - unprivileged_groups_map: Map of unprivileged groups for sensitive attributes.
    #   - Returns: A dictionary containing performance and fairness metrics.
    #   - Internally uses `classification_report` and `aif360.metrics.ClassificationMetric`.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Baseline Model Evaluation and Bias Identification
    # This cell calls `evaluate_and_identify_bias` for the `baseline_model` and `dataset_test_aif`.
    # It prints the classification report and the calculated fairness metrics for 'Gender' and 'SES_Level'.
    # A confusion matrix for the baseline model will also be generated using `seaborn.heatmap`.
    ```

*   **Markdown Cell:**
    Interprets the results, highlighting both general model performance and specific biases. It discusses which sensitive groups are disproportionately affected and by which fairness metrics. For instance, a negative Statistical Parity Difference for 'Gender' might indicate that 'Female' students are less likely to be predicted a favorable outcome than 'Male' students.

---

### Section 8: Visualization of Baseline Bias

*   **Markdown Cell:**
    This section visualizes the identified biases to make them more tangible. Bar plots are used to compare the `Statistical Parity Difference` and `Equal Opportunity Difference` for both `Gender` and `SES_Level`. A relationship plot (pair plot) is also generated to show correlations between features and sensitive attributes.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definitions for Bias Visualization
    # Defines functions to plot fairness metrics and distributions.
    # Function: plot_fairness_metrics(metrics_data: dict, title: str, metric_keys: list) -> None
    #   - metrics_data: Dictionary containing fairness metric values.
    #   - title: Title of the plot.
    #   - metric_keys: List of metric names to plot.
    #   - Generates bar plots comparing fairness metrics for different sensitive attributes.
    # Function: plot_pair_plot(df: pandas.DataFrame, hue_attribute: str) -> None
    #   - df: DataFrame to plot.
    #   - hue_attribute: Sensitive attribute to color-code plots (e.g., 'Gender').
    #   - Generates a `seaborn.pairplot` to show relationships between features and the sensitive attribute.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Bias Visualization
    # This cell calls the plotting functions to visualize baseline biases.
    # It generates bar plots for Statistical Parity Difference and Equal Opportunity Difference for both sensitive attributes.
    # It also generates a `seaborn.pairplot` for selected numerical features colored by 'Gender' and then by 'SES_Level'.
    ```

*   **Markdown Cell:**
    Provides detailed explanations of each visualization. It points out specific disparities shown in the plots, reinforcing the numerical findings from the previous section. For instance, "The bar chart clearly shows a negative SPD for 'Female' students, indicating under-prediction of favorable outcomes."

---

### Section 9: Bias Mitigation Strategy: Reweighing (Pre-processing)

*   **Markdown Cell:**
    Introduces pre-processing as a bias mitigation technique, specifically focusing on `Reweighing` from `aif360.algorithms.preprocessing.Reweighing`, as outlined in Section 4.4.1 of the input document [32]. `Reweighing` works by assigning different weights to the training data instances of privileged and unprivileged groups to balance their representation and achieve statistical parity. The objective is to adjust instance weights such that the probability of the favorable outcome is equal across groups.
    The formal definition of Statistical Parity is given by:
    $$
    \text{Statistical Parity} = P(\hat{Y}=+|A=a) - P(\hat{Y}=+|A=\bar{a})
    $$
    where $P(\hat{Y}=+|A=a)$ is the probability of a positive predicted outcome for the protected group $a$, and $P(\hat{Y}=+|A=\bar{a})$ is for the non-protected group $\bar{a}$. Reweighing aims to make this difference close to zero.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Reweighing Mitigation
    # This function applies the Reweighing algorithm to the training dataset.
    # Function: apply_reweighing_mitigation(dataset_train_aif: aif360.datasets.StandardDataset, privileged_groups: list, unprivileged_groups: list) -> tuple
    #   - dataset_train_aif: The AIF360 StandardDataset for training.
    #   - privileged_groups: List of privileged group definitions.
    #   - unprivileged_groups: List of unprivileged group definitions.
    #   - Returns: A tuple (dataset_train_reweighed_aif, reweighed_sample_weights), where the first is the reweighed dataset in AIF360 format,
    #     and the second is a numpy array of sample weights to be used during model training.
    #   - Internally uses `aif360.algorithms.preprocessing.Reweighing`.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Reweighing Mitigation
    # This cell calls `apply_reweighing_mitigation` on `dataset_train_aif` using the previously defined
    # privileged and unprivileged groups. It extracts the reweighed sample weights.
    ```

*   **Markdown Cell:**
    Explains the impact of `Reweighing`, noting that it doesn't change the data features but assigns new weights to instances, effectively rebalancing the influence of different demographic groups during model training.

---

### Section 10: Model Training with Reweighed Data

*   **Markdown Cell:**
    Explains training a new Logistic Regression model using the reweighed training data. The `sample_weights` obtained from the `Reweighing` algorithm are passed to the `fit` method of the Logistic Regression model. This allows the model to learn from the rebalanced dataset, with the expectation of reducing bias.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition (Reusing `train_logistic_regression_model`)
    # This section reuses the `train_logistic_regression_model` function, but this time
    # providing the `reweighed_sample_weights` to the training process.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Mitigated Model Training
    # This cell calls `train_logistic_regression_model` with `X_train`, `y_train`, and `reweighed_sample_weights`.
    # The new trained model is stored in `mitigated_model`.
    ```

*   **Markdown Cell:**
    Confirms that the model has been trained with the bias mitigation technique applied, ready for re-evaluation of its performance and fairness.

---

### Section 11: Evaluation of Mitigated Model and Fairness Metrics

*   **Markdown Cell:**
    Evaluates the performance and fairness of the `mitigated_model`. Similar to the baseline evaluation, it computes standard classification metrics and fairness metrics (`Statistical Parity Difference`, `Equal Opportunity Difference`, etc.) using `aif360.metrics.ClassificationMetric`. The primary goal is to assess whether the mitigation technique successfully reduced bias without significantly compromising accuracy.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition (Reusing `evaluate_and_identify_bias`)
    # This section reuses the `evaluate_and_identify_bias` function to evaluate the mitigated model.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Mitigated Model Evaluation and Bias Identification
    # This cell calls `evaluate_and_identify_bias` for the `mitigated_model` and `dataset_test_aif`.
    # It prints the classification report and the calculated fairness metrics for 'Gender' and 'SES_Level'.
    # A confusion matrix for the mitigated model will also be generated using `seaborn.heatmap`.
    ```

*   **Markdown Cell:**
    Compares the mitigated model's performance and fairness metrics directly with the baseline results. It highlights improvements in fairness (e.g., SPD closer to zero) and discusses any observed trade-offs with accuracy or other performance metrics.

---

### Section 12: Visualization of Mitigated Bias

*   **Markdown Cell:**
    Visualizes the fairness metrics of the mitigated model and provides a direct comparison with the baseline. This section uses comparative bar plots to clearly show the reduction in bias.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Fairness Metrics Comparison Plot
    # This function creates a bar plot to compare specific fairness metrics between baseline and mitigated models.
    # Function: plot_fairness_metrics_comparison(baseline_metrics: dict, mitigated_metrics: dict, metric_name: str, sensitive_attribute: str) -> None
    #   - baseline_metrics: Dictionary of fairness metrics for the baseline model.
    #   - mitigated_metrics: Dictionary of fairness metrics for the mitigated model.
    #   - metric_name: The specific fairness metric to compare (e.g., 'Statistical Parity Difference').
    #   - sensitive_attribute: The sensitive attribute for which to compare the metric.
    #   - Generates comparative bar plots using `matplotlib.pyplot` and `seaborn`.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Bias Comparison Visualization
    # This cell calls `plot_fairness_metrics_comparison` for 'Statistical Parity Difference'
    # and 'Equal Opportunity Difference' for both 'Gender' and 'SES_Level'.
    ```

*   **Markdown Cell:**
    Interprets the comparison plots, demonstrating how the `Reweighing` technique effectively reduced bias. It quantifies the improvement in fairness metrics and discusses the visual evidence of a more equitable model.

---

### Section 13: Exploring Group Fairness: Equal Opportunity

*   **Markdown Cell:**
    Delves deeper into Group Fairness by focusing on the `Equal Opportunity Difference` metric, as introduced in Section 4.3.2 of the provided document. This metric is crucial for scenarios where fairness for positive outcomes is paramount, ensuring that members of both privileged and unprivileged groups who *actually* belong to the positive class have an equal chance of being correctly classified as positive.
    The `Equal Opportunity Difference` (EOD) is defined as the difference in True Positive Rates (TPR) between the unprivileged and privileged groups, given a positive actual outcome:
    $$
    \text{EOD} = |P(\hat{Y}=+|D=\text{unprivileged}, Y=+) - P(\hat{Y}=+|D=\text{privileged}, Y=+)|
    $$
    A value of 0 indicates perfect equal opportunity.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition (Reusing `calculate_fairness_metrics`)
    # This section reuses the `calculate_fairness_metrics` function, which already computes EOD.
    # No new function definition is required here.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Equal Opportunity Analysis
    # This cell extracts and displays the 'Equal opportunity difference' for both baseline and mitigated models
    # across 'Gender' and 'SES_Level' from the previously computed metrics.
    ```

*   **Markdown Cell:**
    Compares the `Equal Opportunity Difference` for the baseline and mitigated models, discussing the implications for fairness in "favorable" predictions. It explains whether the mitigation strategy improved equal opportunity and what that means for different student groups.

---

### Section 14: Exploring Group Fairness: Predictive Parity

*   **Markdown Cell:**
    Further explores Group Fairness by examining the `Predictive Parity Difference` metric, corresponding to `Predictive Parity` described in Section 4.3.2 of the input document [32]. This metric focuses on the precision of positive predictions, ensuring that when the model predicts a favorable outcome, the probability of that prediction being correct is similar across different groups.
    `Predictive Parity Difference` (PPD) is defined as the difference in Positive Predictive Values (PPV) between the unprivileged and privileged groups:
    $$
    \text{PPD} = |P(Y=+| \hat{Y}=+, D=\text{unprivileged}) - P(Y=+| \hat{Y}=+, D=\text{privileged})|
    $$
    A value of 0 indicates perfect predictive parity.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition (Reusing `calculate_fairness_metrics`)
    # This section reuses the `calculate_fairness_metrics` function, which can be extended to compute PPD,
    # or it can be manually computed from the confusion matrix elements.
    # For AIF360, this is often derived from the `ClassificationMetric` object.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Predictive Parity Analysis
    # This cell calculates and displays the 'Predictive parity difference' for both baseline and mitigated models
    # across 'Gender' and 'SES_Level', extracted or derived from the previously computed metrics.
    ```

*   **Markdown Cell:**
    Compares the `Predictive Parity Difference` for the baseline and mitigated models. It discusses how this metric reflects the reliability of positive predictions across groups and whether the mitigation improved this aspect of fairness.

---

### Section 15: Fairness Bonded Utility (FBU) Concept

*   **Markdown Cell:**
    Introduces the `Fairness Bonded Utility (FBU)` framework, as detailed in Section 4.3.3 of the provided document [32]. FBU provides a conceptual framework for understanding the trade-off between fairness and model performance. It categorizes bias mitigation techniques into five effectiveness levels: *Jointly Advantageous*, *Impressive*, *Reversed*, *Deficient*, and *Jointly Disadvantageous*. This section explains these levels and the idea of a 2D coordinate system where mitigation strategies are plotted based on their impact on fairness and performance relative to a "trade-off baseline."

*   **Code Cell:**
    ```python
    # Code Cell: Conceptual Outline of FBU Calculation Steps
    # This cell describes the high-level steps involved in conceptualizing FBU without direct code implementation,
    # as FBU itself is a framework for evaluating techniques, rather than a single algorithm.
    # 1. Train an original model and establish baseline fairness and performance.
    # 2. Generate "pseudo-models" (N_beta) by systematically replacing a percentage (beta) of the original model's predictions
    #    with a constant output label (e.g., always predicting '0' or '1') to simulate varying degrees of fairness intervention.
    # 3. Calculate fairness and performance for each pseudo-model.
    # 4. Define a "trade-off baseline" by connecting the original model's point with the pseudo-model points.
    # 5. Apply different bias mitigation techniques (like Reweighing).
    # 6. Plot the fairness and performance of these techniques on the 2D coordinate system relative to the baseline.
    # 7. Categorize techniques into the five FBU effectiveness levels.
    ```

*   **Code Cell:**
    *(No direct execution of FBU visualization for this notebook due to its conceptual nature and complexity)*

*   **Markdown Cell:**
    Explains how FBU helps in making informed decisions about which bias mitigation strategies are most suitable, considering the desired balance between fairness and model utility. It emphasizes that our `Reweighing` example would ideally fall into the "Impressive" or "Jointly Advantageous" regions if it improves both fairness and maintains or improves performance.

---

### Section 16: Interactive Exploration of Fairness vs. Accuracy Trade-off

*   **Markdown Cell:**
    This section provides an interactive tool for learners to explore the trade-off between model fairness and accuracy. Users can adjust a prediction threshold (or a parameter of a mitigation strategy) using sliders or dropdowns, and observe its real-time impact on selected fairness metrics and accuracy. This fulfills the "user interaction" requirement and allows for deeper understanding of the complexities of bias mitigation.

*   **Code Cell:**
    ```python
    # Code Cell: Function Definition for Interactive Fairness-Accuracy Trade-off Plot
    # This function generates an interactive plot where a slider or dropdown controls a parameter
    # (e.g., prediction threshold) and updates fairness and accuracy metrics.
    # Function: interactive_fairness_accuracy_plot(model: sklearn.linear_model.LogisticRegression, dataset_test_aif: aif360.datasets.StandardDataset, privileged_groups: list, unprivileged_groups: list, metric_to_plot: str) -> None
    #   - model: The trained classification model (e.g., mitigated_model).
    #   - dataset_test_aif: AIF360 StandardDataset for the test set.
    #   - privileged_groups: List of privileged group definitions.
    #   - unprivileged_groups: List of unprivileged group definitions.
    #   - metric_to_plot: The specific fairness metric to display (e.g., 'Statistical Parity Difference').
    #   - Generates an interactive plot using `ipywidgets` for a slider/dropdown and `matplotlib.pyplot` for plotting.
    #   - Each interactive control will include inline help text describing its purpose.
    ```

*   **Code Cell:**
    ```python
    # Code Cell: Execute Interactive Trade-off Plot
    # This cell calls `interactive_fairness_accuracy_plot` for the `mitigated_model`,
    # allowing users to interactively adjust the prediction threshold and visualize
    # the changes in accuracy and 'Statistical Parity Difference' for 'Gender'.
    ```

*   **Markdown Cell:**
    Explains how to use the interactive controls and interprets the observed trade-offs. It emphasizes that achieving perfect fairness often comes with some cost to overall accuracy, and educators must decide on an acceptable balance based on ethical considerations.

---

### Section 17: Conclusion

*   **Markdown Cell:**
    Summarizes the key insights gained from the notebook: the prevalence of bias in AI systems for education, the use of fairness metrics for identification, and the effectiveness (and trade-offs) of pre-processing mitigation techniques like `Reweighing`. It reiterates the critical need for medical educators to consider AI fairness in curriculum design to ensure equitable and inclusive learning experiences for all students, aligning with the learning outcomes.

*   **Code Cell:**
    *(No code in this section)*

*   **Code Cell:**
    *(No execution in this section)*

*   **Markdown Cell:**
    *(No explanation for execution)*

---

### Section 18: References

*   **Markdown Cell:**
    Lists all external references cited within the notebook, including the foundational papers [32] and [33], as well as any other sources for datasets or libraries used.

    **References:**
    *   [32] Chinta, S.V., Wang, Z., Yin, Z., Hoang, N., Gonzalez, M., Quy, T.L., Zhang, W.: Fairaied: Navigating fairness, bias, and ethics in educational ai applications. (2024).
    *   [33] Chinta, S.V., Wang, Z., Zhang, X., Doan, T., Kashif, A., Smith, M.A., Zhang, W.: Ai-driven healthcare: A survey on ensuring fairness and mitigating bias (2024).
    *   Scikit-learn documentation: https://scikit-learn.org/
    *   AIF360 documentation: https://aif360.readthedocs.io/
    *   Matplotlib documentation: https://matplotlib.org/
    *   Seaborn documentation: https://seaborn.pydata.org/
    *   IPyWidgets documentation: https://ipywidgets.readthedocs.io/
    *   Pandas documentation: https://pandas.pydata.org/
    *   NumPy documentation: https://numpy.org/

*   **Code Cell:**
    *(No code in this section)*

*   **Code Cell:**
    *(No execution in this section)*

*   **Markdown Cell:**
    *(No explanation for execution)*

