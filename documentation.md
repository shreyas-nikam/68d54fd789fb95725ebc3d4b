id: 68d54fd789fb95725ebc3d4b_documentation
summary: FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications

## Introduction to FairAIED
Duration: 0:05:00

Welcome to the **FairAIED** lab! This codelab provides a comprehensive guide to understanding, developing, and deploying AI applications in education, with a particular focus on ensuring fairness, mitigating bias, and adhering to ethical principles. Artificial Intelligence holds immense promise for revolutionizing education, from personalized learning paths to intelligent tutoring systems and, as explored in this application, course recommendation engines. However, like any powerful technology, AI is not without its pitfalls. If not carefully designed and monitored, AI systems can inadvertently perpetuate or even amplify existing societal biases, leading to unfair or discriminatory outcomes for certain student populations.

This interactive Streamlit application, **FairAIED**, is designed to be a hands-on environment for developers, researchers, and educators to:

*   **Understand Recommendation Systems**: Delve into the fundamental mechanisms of how AI-driven recommendation systems function and how biases can creep into their decision-making processes, especially when relying on historical data.
*   **Explore Fairness Metrics**: Familiarize yourself with a suite of quantitative fairness metrics such as Demographic Parity, Equal Opportunity, Equalized Odds, and Predictive Parity. Understanding these metrics is crucial for objectively evaluating the fairness of an AI model's outputs.
*   **Investigate Fairness Techniques**: Discover conceptual approaches and practical techniques to incorporate fairness constraints directly into the design and training of recommendation algorithms, aiming for more equitable outcomes.
*   **Analyze Trade-offs**: Recognize the inherent challenges and trade-offs that often exist between achieving high model accuracy (utility), ensuring fairness, and promoting diversity in recommendations.

The application leverages a synthetic data generation mechanism, allowing you to experiment with controlled biases and observe their impact on various fairness metrics and model behaviors. By engaging with this lab, you will gain a robust understanding of the critical considerations required to build responsible and ethical AI solutions for the educational sector.

<aside class="positive">
<b>Key Takeaway:</b> FairAIED emphasizes the importance of proactive and continuous evaluation of AI systems in education to prevent harm and ensure equitable access to opportunities for all students.
</aside>

## Application Architecture and Setup
Duration: 0:03:00

The FairAIED application is built using Streamlit, a popular Python library for creating interactive web applications with minimal code. The application is structured into a main entry point (`app.py`) and several modular pages (`application_pages/page1.py`, `page2.py`, `page3.py`), enhancing maintainability and readability.

### Application Flow
The core `app.py` file sets up the Streamlit page configuration, displays the main title, and serves as the navigation hub. It defines global parameters for data generation (Number of Students, Number of Courses, Bias Strength) using sidebar sliders, which persist across different pages via Streamlit's session state. Based on the user's selection in the sidebar navigation, `app.py` dynamically loads and renders the content from the respective page modules.

```python
# app.py snippet for navigation
page = st.sidebar.selectbox(label="Navigation", options=["Data Generation & Exploration", "Fairness Notions & Metrics", "Ethical Approaches & Future Directions"])
if page == "Data Generation & Exploration":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Fairness Notions & Metrics":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Ethical Approaches & Future Directions":
    from application_pages.page3 import run_page3
    run_page3()
```

### Global Data Generation Parameters

The sidebar in `app.py` features interactive sliders that control the synthetic dataset generation:

*   **Number of Students**: Adjusts the total count of synthetic student profiles.
*   **Number of Courses**: Adjusts the total count of synthetic courses.
*   **Bias Strength**: A crucial parameter that controls the strength of correlation between a student's `major` and a course's `subject_area` during interaction generation. A higher value here will amplify the inherent bias, making it more apparent in the generated interaction scores.

<aside class="positive">
<b>Tip:</b> Experiment with the `Bias Strength` slider. You'll observe its direct impact on the interaction patterns visualized in the "Data Generation & Exploration" page. This allows for a direct understanding of how biases can be simulated and observed.
</aside>

### How to Run the Application

To run this Streamlit application locally, ensure you have Python installed and Streamlit is in your environment.

1.  **Save the files:** Create the following file structure on your machine:
    ```
    your_project_folder/
    ├── app.py
    └── application_pages/
        ├── __init__.py
        ├── page1.py
        ├── page2.py
        └── page3.py
    ```
    Copy the provided code into `app.py`, `application_pages/page1.py`, `application_pages/page2.py`, and `application_pages/page3.py` respectively. Ensure `__init__.py` is an empty file to make `application_pages` a Python package.

2.  **Install dependencies:**
    Open your terminal or command prompt and navigate to `your_project_folder`. Install the necessary libraries:
    ```bash
    pip install streamlit pandas numpy plotly
    ```

3.  **Run the application:**
    From `your_project_folder`, execute the following command:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

## Data Generation and Exploration
Duration: 0:10:00

This section, powered by `application_pages/page1.py`, focuses on creating and understanding the synthetic dataset that simulates an educational environment. This synthetic data is crucial because it allows us to introduce and control biases, making their effects observable and measurable for our fairness analysis.

### Synthetic Data Generation Explained

The `generate_synthetic_data` function is the heart of this page. It produces three interconnected Pandas DataFrames: `students_df`, `courses_df`, and `interactions_df`.

<aside class="positive">
<b>Insight:</b> The use of `@st.cache_data` on `generate_synthetic_data` is a Streamlit optimization. It ensures that the data generation function runs only when its input parameters (number of students, courses, bias strength) change, preventing unnecessary re-computation and speeding up the application.
</aside>

#### 1. `students_df` (Student Profiles)

This DataFrame contains simulated student profiles with the following columns:

*   `student_id` (int): A unique identifier for each student.
*   `gpa` (float): The Grade Point Average, a numeric value between 2.0 and 4.0.
*   `major` (category): The student's declared major (e.g., 'Computer Science', 'Literature'). This is a **sensitive attribute** that will be used for introducing and analyzing bias.

#### 2. `courses_df` (Course Information)

This DataFrame holds information about the available courses:

*   `course_id` (int): A unique identifier for each course.
*   `difficulty` (category): The perceived difficulty ('Easy', 'Medium', 'Hard').
*   `subject_area` (category): The broad subject category (e.g., 'STEM', 'Humanities', 'Business'). This is linked to the `major` to introduce bias in interactions.

#### 3. `interactions_df` (Student-Course Interactions)

This DataFrame records the simulated engagement between students and courses:

*   `student_id` (int): Foreign key referencing `students_df`.
*   `course_id` (int): Foreign key referencing `courses_df`.
*   `interaction_score` (float): A normalized score (0.0 to 1.0) representing interest or engagement. This score is directly influenced by the `bias_strength` parameter.

    The bias is introduced by giving a positive boost to `interaction_score` if a student's `major` aligns with a course's `subject_area` (e.g., a "Computer Science" student with a "STEM" course). Conversely, a slight negative adjustment is applied for misaligned pairs. This simulates real-world scenarios where students naturally gravitate towards courses related to their primary field of study, and where historical data might reflect this preference.

    ```python
    # Snippet from generate_synthetic_data demonstrating bias
    if major_to_subject.get(student_major) == course_subject:
        # Positive bias: student's major aligns with course subject
        score_adjustment = bias_strength * 0.3
        interaction_score = base_score + score_adjustment
    else:
        # Negative bias: student's major does not align
        score_adjustment = bias_strength * 0.15
        interaction_score = base_score - score_adjustment
    ```

#### Data Relationships Diagram

The three DataFrames are interconnected as follows:

```mermaid
graph TD
    A[students_df] --> B[interactions_df]
    C[courses_df] --> B
    B[interactions_df]: student_id, course_id, interaction_score
    A[students_df]: student_id (PK), gpa, major
    C[courses_df]: course_id (PK), difficulty, subject_area
```

*   `student_id` is the primary key for `students_df` and a foreign key in `interactions_df`.
*   `course_id` is the primary key for `courses_df` and a foreign key in `interactions_df`.
*   The `major` from `students_df` and `subject_area` from `courses_df` are used to induce bias in the `interaction_score` within `interactions_df`.

### Exploring the Generated Data

After generating the synthetic data, the application displays the first 5 rows of each DataFrame and provides several visualizations to help you understand the data's characteristics and the impact of the introduced bias.

*   **Distribution of Student GPAs**: A histogram showing the spread of academic performance among students.
*   **Distribution of Student Majors**: A bar chart illustrating the number of students in each major, representing the demographic composition.
*   **Distribution of Course Difficulties**: A bar chart showing the count of courses per difficulty level.
*   **Distribution of Course Subject Areas**: A bar chart indicating the number of courses available in each subject area.

The most crucial visualization for understanding the introduced bias is the:

*   **Average Interaction Score by Major and Course Subject Area (Heatmap)**: This heatmap shows the average interaction score for each combination of student major and course subject area. As you adjust the `Bias Strength` slider in the sidebar, you will observe the heatmap colors change: darker (higher scores) where majors align with subject areas (e.g., 'Computer Science' students with 'STEM' courses), and lighter (lower scores) where they do not. This visual cue directly demonstrates how the synthetic bias influences student preferences.

<aside class="negative">
<b>Warning:</b> If you set the `Number of Students` or `Number of Courses` to 0, or if the `Bias Strength` is very low with a small number of interactions, some visualizations might show "No data available" or appear uniform. Increase these parameters to see more varied and meaningful plots.
</aside>

## Fairness Notions and Metrics
Duration: 0:15:00

This section, managed by `application_pages/page2.py`, dives deep into the theoretical underpinnings of fairness in AI, establishing the notations and definitions critical for evaluating AI systems in educational settings.

### Notations for Fairness Analysis

To precisely discuss fairness, a set of common notations is introduced:

| Notation | Definition |
| :- | : |
| $Y$ | Class attribute with binary values $\{+,-\}$, representing positive and negative outcomes. |
| $A$ | Binary sensitive attribute with values $\{a, \bar{a}\}$, indicating membership in a protected or non-protected group. |
| $a$ | Protected group, safe against discrimination by fairness constraints. |
| $\bar{a}$ | Non-protected group, important for analysis. |
| $\hat{Y}$ | Predicted outcome by the model, approximating the true class attribute $Y$. |
| TP | True Positive, model correctly predicts a positive outcome. |
| FN | False Negative, model incorrectly predicts a negative outcome for a positive case. |
| TN | True Negative, model correctly predicts a negative outcome. |
| FP | False Positive, model incorrectly predicts a positive outcome for a negative case. |
| $f(\cdot)$ | Predictive model function. |
| $dY(\cdot)$ | Distance metric in output space. |
| $dX(\cdot)$ | Distance metric in input space. |
| $L$ | Lipschitz constant. |
| $y_i$ | Probabilistic classification output for instance $x_i$. |
| $x_j$ | Top $k$-nearest neighbors for $x_i$. |

These notations are foundational for formalizing and quantifying different fairness objectives.

### Fairness Notions: Individual vs. Group Fairness

Fairness in AI is typically approached from two perspectives:

*   **Individual Fairness**: This principle dictates that **similar individuals should be treated similarly**. In an educational context, this means two students with identical qualifications and backgrounds (excluding sensitive attributes) should receive the same course recommendations or assessments. It aims to prevent discrimination against specific individuals.
*   **Group Fairness**: This principle focuses on ensuring **equitable outcomes across predefined demographic groups**. It seeks to prevent systematic biases that might disadvantage one group (e.g., students from a particular major or socioeconomic background) compared to another.

### Fairness Evaluating Resources (Metrics)

Evaluating fairness requires concrete metrics that quantify how well an AI system adheres to individual or group fairness principles.

#### Individual Fairness Metrics

Individual fairness often involves distance-based or ranking-based approaches:

*   **Distance-Based Individual Fairness (Lipschitz Condition)**: Adheres to the Lipschitz condition, implying that small changes in input should lead to only small changes in output.
    $$ dY(f(x_i), f(x_j)) \leq L \cdot dX(x_i, x_j) $$
    Another metric in classification measures the average output distance between an individual and their $k$-nearest neighbors:
    $$ 1 - \frac{1}{n \cdot k} \sum_{i=1}^{n} \left| y_i - \sum_{j=1}^{k} \overline{Y_j} \right| $$
*   **Ranking-Based Individual Fairness**: Relevant for recommendation systems, this can be evaluated using metrics like Normalized Discounted Cumulative Gain (NDCG@k) and Expected Reciprocal Rank (ERR@k), which assess the quality of ranked recommendations for individuals.

#### Group Fairness Metrics

Group fairness metrics compare outcomes across different protected and non-protected groups.

*   **Statistical Parity (Demographic Parity)**: Requires that the proportion of positive outcomes is roughly equal across different demographic groups.
    $$ P(\hat{Y}|A = a) - P(\hat{Y}|A = \bar{a}) \leq \epsilon $$
    Quantified as:
    $$ \text{Statistical Parity} = P(\hat{Y} = +|A = a) - P(\hat{Y} = +|A = \bar{a}) $$
*   **Equal Opportunity**: Focuses on fairness for positive outcomes when the true outcome is positive, requiring equal True Positive Rates (TPR) across groups.
    $$ P(\hat{Y} = +|A = a, Y = +) = P(\hat{Y} = +|A = \bar{a}, Y = +) $$
    Where:
    $$ \text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}} $$
    And quantified by the absolute difference in False Negative Rates (FNR):
    $$ \text{Equal Opportunity} = \left| P(\hat{Y} = -|Y = +, A = a) - P(\hat{Y} = -|Y = +, A = \bar{a}) \right| $$
*   **Equalized Odds**: A stronger notion requiring both True Positive Rates and False Positive Rates to be equal across groups, given the true outcome.
    $$ P(\hat{Y} = +|A = a, Y = y) = P(\hat{Y} = +|A = \bar{a}, Y = y), \quad \text{where } y \in \{+,-\} $$
    Quantified as:
    $$ \text{Equalized Odds} = \sum_{y \in \{+,-\}} \left| P(\hat{Y} = +|A = a, Y = y) - P(\hat{Y} = +|A = \bar{a}, Y = y) \right| $$
*   **Predictive Parity (Positive Predictive Value Parity)**: Requires that both groups have an equal Positive Predictive Value (PPV), focusing on the precision of positive predictions.
    $$ \text{PPV} = \frac{\text{TP}}{\text{TP}+\text{FP}} $$
    Formally:
    $$ P(Y = +|\hat{Y} = +, A = a) = P(Y = +|\hat{Y} = +, A = \bar{a}) $$
    Quantified by:
    $$ \text{Predictive Parity} = \left| P(Y = +|\hat{Y} = +, A = a) - P(Y = +|\hat{Y} = +, A = \bar{a}) \right| $$
*   **Predictive Equality (False Positive Rate Balance)**: Concerned with ensuring equal False Positive Rates (FPR) across groups.
    $$ P(\hat{Y} = +|Y = -, A = a) = P(\hat{Y} = +|Y = -, A = \bar{a}) $$
    Quantified by:
    $$ \text{Predictive Equality} = \left| P(\hat{Y} = +|Y = -, A = a) - P(\hat{Y} = +|Y = -, A = \bar{a}) \right| $$

### Fairness Bonded Utility (FBU)

FBU is a framework for analyzing the trade-off between model utility (performance) and fairness. It categorizes fairness-enhancing techniques into five effectiveness levels based on their impact on both dimensions:

*   **Level 1: Jointly Advantageous** (Improves both)
*   **Level 2: Impressive** (Significantly improves one with minimal cost to the other)
*   **Level 3: Reversed** (Clear trade-off, notable cost to one for improvement in the other)
*   **Level 4: Deficient** (Performs poorly on both)
*   **Level 5: Jointly Disadvantageous** (Actively worsens both)

<aside class="positive">
<b>Important:</b> No single fairness metric is universally "best." The most appropriate metric depends on the specific context, the potential harms, and the ethical priorities of the application. Often, trade-offs between different fairness metrics exist, and improving one might negatively impact another.
</aside>

## Ethical Approaches and Future Directions
Duration: 0:17:00

This final section, implemented in `application_pages/page3.py`, extends beyond quantitative metrics to explore practical approaches for improving fairness, address broader ethical considerations in AIED, and look at the future landscape of responsible AI development.

### Approaches to Improve Fairness

Bias mitigation techniques can be applied at different stages of the machine learning pipeline:

1.  **Pre-processing Techniques**: Modify the input data before model training.
    *   **Re-weighing**: Assigns different weights to data points or groups to balance their influence.
    *   **Resampling**: Alters data distribution by oversampling underrepresented groups or undersampling overrepresented ones.

2.  **In-processing Techniques**: Integrate fairness constraints directly into the model training algorithm.
    *   **Regularization Methods**: Add fairness-related terms to the loss function to penalize unfairness.
    *   **Constraint-based Optimization**: Formulate fairness objectives as explicit constraints during model optimization.
    *   **Adversarial Debiasing**: Uses an adversarial network to learn representations that are independent of sensitive attributes.

3.  **Post-processing Techniques**: Adjust model predictions after training without altering the model or data.
    *   **Threshold Adjustment**: Modifies classification thresholds for different groups to achieve fairness.
    *   **Outcome Perturbation**: Randomly flips some predictions for certain groups to meet fairness criteria.

### Ethical Considerations and Frameworks in AI for Education

Beyond technical fairness, the deployment of AI in education presents a range of ethical challenges.

| Ethical Consideration        | Description                                                                                             |
| : | : |
| Bias and Discrimination      | AI systems inheriting and amplifying biases, leading to discriminatory treatment of student groups.     |
| Privacy and Data Security    | Handling sensitive educational data, concerns about collection, storage, usage, and breaches.           |
| Autonomy and Agency          | Potential reduction of student and teacher autonomy due to over-reliance on AI recommendations.         |
| Transparency and Explainability | Difficulty in understanding 'why' an AI system made a particular recommendation or assessment.      |
| Accountability               | Challenges in establishing responsibility for flawed or harmful AI decisions.                         |
| Digital Divide               | Exacerbation of inequalities for students lacking access to technology and internet required by AIED systems. |

Numerous ethical guidelines and frameworks emphasize principles like Fairness, Accountability, Transparency, Privacy, Safety, Beneficence, and Non-maleficence to guide responsible AIED development.

### Tools and Datasets for Fairness Research

The field is supported by a growing ecosystem of open-source tools and datasets:

| Category  | Name                            | Description                                                                                                                                                                                                                                                            |
| :-- | : | : |
| Tools     | AI Fairness 360 (AIF360)        | IBM's extensible open-source toolkit for fairness metrics and bias mitigation.                                                                                                                                                                                       |
| Tools     | Fairlearn                       | Microsoft's open-source toolkit to assess and improve model fairness, integrates with scikit-learn.                                                                                                                                                                    |
| Tools     | TensorFlow Fairness Indicators  | Tools on TensorFlow Extended (TFX) for computing fairness metrics and visualizations.                                                                                                                                                                                |
| Tools     | What-If Tool (WIT)              | Interactive visual tool to probe model behavior and compare performance across data slices.                                                                                                                                                                          |
| Tools     | FairTest                        | Framework for automatically testing machine learning models for implicit bias.                                                                                                                                                                                         |
| Tools     | FairML                          | Python package to audit models for fairness by quantifying feature contributions to predictions.                                                                                                                                                                     |
| Tools     | Aequitas                        | Open-source bias audit toolkit from University of Chicago for assessing fairness across multiple dimensions.                                                                                                                                                         |
| Datasets  | TIMSS                           | International comparative data on student achievement in mathematics and science.                                                                                                                                                                                      |
| Datasets  | PISA                            | Assesses 15-year-old students' reading, mathematics, and science literacy.                                                                                                                                                                                             |
| Datasets  | NELS:88                         | Longitudinal study tracking students from 8th grade through post-secondary experiences.                                                                                                                                                                                |
| Datasets  | EdNet                           | Large-scale dataset for educational purposes, student interaction data with online learning platforms.                                                                                                                                                               |
| Datasets  | KDDCUP 2015                     | Dataset from an educational data mining competition focused on predicting student dropout.                                                                                                                                                                             |
| Datasets  | OULAD                           | Demographic, activity, and assessment data from students in higher education.                                                                                                                                                                                          |
| Datasets  | ASSISTments                     | Widely used dataset from an online tutoring platform for mathematics.                                                                                                                                                                                                  |
| Datasets  | Statlog (German Credit Data)    | Classic dataset for fairness concerns in credit scoring, with parallels in educational resource allocation.                                                                                                                                                            |

### Challenges and Future Directions

The journey towards fair and ethical AI in education is ongoing, with several challenges and promising future directions:

*   **Personalization Versus Data Privacy**: Balancing granular data for personalization with the imperative to protect sensitive student information.
*   **Data-Related Challenges**: Addressing lack of representative data and ensuring ethical data collection.
*   **The Definition and Measurement of Fairness**: Navigating conflicting fairness definitions and choosing appropriate metrics.
*   **Trade-offs Between Fairness and Accuracy**: Managing the tension where improving fairness might reduce overall model accuracy.
*   **Digital Divide**: Preventing AIED systems from exacerbating inequalities for digitally disadvantaged students.
*   **Teacher and Student Adoption**: Building trust through transparent, explainable, and user-friendly systems.

Future directions include:

*   **Human-in-the-Loop AI**: Integrating human educators and students into AI decision-making processes.
*   **Explainable AI (XAI) for Fairness**: Developing AIED systems that can clearly explain their fairness implications.
*   **Contextual Fairness**: Adapting fairness definitions to specific educational settings and goals.
*   **Intersectional Fairness**: Addressing biases arising from the intersection of multiple sensitive attributes.
*   **Proactive Fairness Design**: Integrating fairness considerations from the initial design phase of AIED systems.
*   **Policy and Governance**: Establishing robust regulations for ethical AI deployment in education.

<aside class="positive">
<b>Conclusion:</b> By continuously engaging with these challenges and exploring these future directions, the field of AI in education can strive to build systems that are not only intelligent and effective but also fundamentally fair and equitable for all learners.
</aside>

## Conclusion
Duration: 0:02:00

Congratulations! You have successfully navigated the **FairAIED** codelab, exploring the critical aspects of fairness, bias, and ethics in educational AI applications. You've gained insights into:

*   **Synthetic Data Generation**: Understanding how to create controlled datasets to simulate and analyze bias.
*   **Data Exploration**: Visualizing and interpreting data characteristics, especially the impact of introduced biases.
*   **Fairness Notions**: Differentiating between individual and group fairness principles.
*   **Fairness Metrics**: Familiarizing yourself with key quantitative metrics like Statistical Parity, Equal Opportunity, and Equalized Odds, and their mathematical formulations.
*   **Bias Mitigation Approaches**: Learning about pre-processing, in-processing, and post-processing techniques to enhance fairness.
*   **Ethical Considerations**: Recognizing the broader ethical dilemmas in AIED and the frameworks designed to address them.
*   **Tools and Resources**: Becoming aware of available open-source tools and datasets for further research and development in fair AI.

The journey towards truly fair and ethical AI in education is complex and ongoing. This lab serves as a foundational step, equipping you with the knowledge and context to critically evaluate, design, and implement AI solutions that are not only intelligent but also equitable and responsible. We encourage you to continue experimenting with the application's parameters, explore the provided resources, and delve deeper into the fascinating and crucial field of AI fairness.

Thank you for participating in the FairAIED lab!

## References

The concepts and discussions in this lab are informed by contemporary research in the field of fairness, accountability, and transparency in AI. Key references include:

*   [32] (Placeholder for a specific reference like FairAIE or others mentioned in the original notebook that might have specific citations)
*   Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on fairness in recommender systems. *ACM Computing Surveys (CSUR)*, 54(4), 1-35.
*   Barocas, S., Hardt, M., & Narayanan, A. (2017). *Fairness and machine learning: Limitations and opportunities*. Online.
*   Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction. *arXiv preprint arXiv:1703.00056*.
*   Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In *Proceedings of the 3rd innovations in theoretical computer science conference* (pp. 214-223).
*   Verma, S., & Rubin, J. (2018). Fairness definitions explained. In *Proceedings of the 2018 IEEE/ACM international workshop on software fairness* (pp. 1-7).
*   A. Narayanan, Fairness and machine learning. (2020). *https://fairmlbook.org/*
*   (Other relevant academic papers, books, or online resources)
