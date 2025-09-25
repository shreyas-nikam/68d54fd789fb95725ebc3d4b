
import streamlit as st
import pandas as pd

def run_page3():
    st.header("Ethical Approaches & Future Directions")

    st.markdown("""
    ## 4.4 Approaches to Improve Fairness

    Addressing biases and enhancing fairness in AI systems, especially in educational applications, involves a multi-faceted approach. These methods can be broadly categorized based on the stage of the machine learning pipeline at which the intervention occurs: Pre-processing, In-processing, and Post-processing.

    ### Pre-processing Techniques

    These techniques aim to mitigate bias by transforming the input data before it is fed into the machine learning model. The goal is to create a more balanced and fair dataset.

    *   **Re-weighing:** Assigns different weights to individual training examples or groups to equalize their representation or impact on the model's learning process. For example, instances from underrepresented protected groups might be given higher weights.
    *   **Resampling:** Modifies the distribution of the training data.
        *   **Oversampling:** Duplicates instances from underrepresented groups to increase their presence in the dataset.
        *   **Undersampling:** Reduces the number of instances from overrepresented groups.

    ### In-processing Techniques

    These methods incorporate fairness constraints directly into the model training process, modifying the learning algorithm itself to produce fairer outcomes.

    *   **Regularization Methods:** Add fairness-related terms to the model's loss function, penalizing models that exhibit unfairness. This encourages the model to learn representations and decision boundaries that are more equitable.
    *   **Constraint-based Optimization:** Formulates fairness objectives as explicit constraints during the model's optimization, ensuring that the final model adheres to specific fairness criteria while maximizing utility.
    *   **Adversarial Debiasing:** Uses an adversarial learning framework, where a "debiasing" component attempts to remove sensitive attribute information from the learned representations, while the main model tries to perform its primary task.

    ### Post-processing Techniques

    These techniques adjust the model's predictions after training, without modifying the model itself or the training data. They are typically applied to the output of a pre-trained, potentially biased, model.

    *   **Threshold Adjustment:** Modifies the classification thresholds for different demographic groups to achieve a desired fairness metric (e.g., equalizing true positive rates).
    *   **Outcome Perturbation:** Randomly flips some predictions for certain groups to satisfy fairness criteria, often used when other methods are not feasible or when small adjustments are sufficient.

    The choice of approach depends on various factors, including the type of bias, the available data, the computational resources, and the specific fairness goals. Each method has its own trade-offs regarding computational cost, model interpretability, and the extent to which fairness can be improved without significantly compromising utility.

    ---

    ## 5 Ethical Considerations and Frameworks in AI for Education

    The deployment of AI in education is fraught with ethical dilemmas that necessitate careful consideration and robust frameworks. Beyond technical fairness metrics, AIED systems must align with broader societal values and educational principles.

    ### Key Ethical Dilemmas:

    *   **Bias and Discrimination:** As discussed, AI systems can inherit and amplify biases, leading to discriminatory treatment of certain student groups.
    *   **Privacy and Data Security:** Educational data is highly sensitive. AIED systems often require vast amounts of personal and academic data, raising concerns about data collection, storage, usage, and potential breaches.
    *   **Autonomy and Agency:** Over-reliance on AI recommendations might diminish student and teacher autonomy, potentially leading to a prescriptive educational experience rather than a personalized, empowering one.
    *   **Transparency and Explainability:** The "black box" nature of some AI models can make it difficult to understand *why* a particular recommendation or assessment was made, hindering trust and accountability.
    *   **Accountability:** When an AI system makes a flawed or harmful decision, establishing who is responsible (developers, institutions, or the AI itself) can be challenging.
    *   **Digital Divide:** AIED systems often require access to technology and internet, potentially exacerbating inequalities for students lacking these resources.

    ### Ethical Guidelines, Policies, and Frameworks:

    Numerous organizations and governments have proposed guidelines and frameworks to navigate these ethical challenges. These often emphasize principles such as:

    *   **Fairness:** Ensuring equitable treatment and outcomes for all students.
    *   **Accountability:** Clear lines of responsibility for AI system decisions.
    *   **Transparency/Explainability:** Making AI decisions understandable to users.
    *   **Privacy:** Protecting sensitive student data.
    *   **Safety and Robustness:** Ensuring AI systems are reliable and do not cause harm.
    *   **Beneficence:** AI systems should be designed to do good and improve educational outcomes.
    *   **Non-maleficence:** AI systems should avoid causing harm.

    These principles guide the responsible development and deployment of AI in education, fostering trust and ensuring that AI serves as a tool for empowerment rather than a source of new disparities.

    ### Table 2: Ethical Considerations in AI in Education
    """)

    ethical_data = {
        "Ethical Consideration": [
            "Bias and Discrimination",
            "Privacy and Data Security",
            "Autonomy and Agency",
            "Transparency and Explainability",
            "Accountability",
            "Digital Divide"
        ],
        "Description": [
            "AI systems inheriting and amplifying biases, leading to discriminatory treatment of student groups.",
            "Handling sensitive educational data, concerns about collection, storage, usage, and breaches.",
            "Potential reduction of student and teacher autonomy due to over-reliance on AI recommendations.",
            "Difficulty in understanding 'why' an AI system made a particular recommendation or assessment.",
            "Challenges in establishing responsibility for flawed or harmful AI decisions.",
            "Exacerbation of inequalities for students lacking access to technology and internet required by AIED systems."
        ]
    }
    df_ethical = pd.DataFrame(ethical_data)
    st.dataframe(df_ethical, hide_index=True)


    st.markdown("""
    ---

    ## 6 Tools and Datasets

    The field of fair AI is rapidly evolving, supported by a growing ecosystem of tools and publicly available datasets that facilitate research and development in this area.

    ### 6.1 Tools for Fairness Analysis and Mitigation

    Several open-source libraries and platforms have emerged to help practitioners and researchers analyze, understand, and mitigate bias in AI systems:

    *   **AI Fairness 360 (AIF360):** An extensible open-source toolkit developed by IBM that provides a comprehensive set of fairness metrics and bias mitigation algorithms for classification and regression tasks.
    *   **Fairlearn:** A Microsoft-developed open-source toolkit that empowers developers of AI systems to assess and improve the fairness of their models. It integrates with scikit-learn and provides various fairness metrics and mitigation techniques.
    *   **TensorFlow Fairness Indicators:** A collection of tools built on TensorFlow Extended (TFX) for computing fairness metrics and visualizing model performance across different groups in a robust, scalable manner.
    *   **What-If Tool (WIT):** An interactive visual tool that allows users to probe the behavior of machine learning models with minimal code, supporting fairness investigations by comparing model performance across different data slices.
    *   **FairTest:** A framework for automatically testing machine learning models for implicit bias, designed to uncover hidden biases in data and models.
    *   **FairML:** A Python package that offers tools for auditing machine learning models for fairness by quantifying the contribution of individual input features to a model's prediction.
    *   **Aequitas:** An open-source bias audit toolkit for machine learning developed by the Center for Data Science and Public Policy at the University of Chicago. It provides a user-friendly interface to assess fairness across multiple dimensions.

    ### 6.2 Datasets for Fairness Research in Education

    While generic fairness datasets exist, several datasets are particularly relevant for research into fairness in educational AI:

    *   **TIMSS (Trends in International Mathematics and Science Study):** Provides international comparative data on student achievement in mathematics and science at grades 4 and 8.
    *   **PISA (Programme for International Student Assessment):** Assesses 15-year-old students' reading, mathematics, and science literacy every three years.
    *   **NELS:88 (National Education Longitudinal Study of 1988):** A longitudinal study tracking a cohort of students from 8th grade through their post-secondary experiences.
    *   **EdNet:** A large-scale dataset for educational purposes, containing student interaction data with online learning platforms.
    *   **KDDCUP 2015:** A dataset from an educational data mining competition focused on predicting student dropout.
    *   **OULAD (Open University Learning Analytics Dataset):** Contains demographic, activity, and assessment data from students in higher education.
    *   **ASSISTments:** A widely used dataset from an online tutoring platform for mathematics.
    *   **Statlog (German Credit Data):** While not exclusively educational, this is a classic dataset used to demonstrate fairness concerns in credit scoring, which has parallels in educational resource allocation.

    ### Table 3: Tools and Datasets for Fairness in AIED
    """)

    tools_data = {
        "Category": ["Tools", "Tools", "Tools", "Tools", "Tools", "Tools", "Tools",
                     "Datasets", "Datasets", "Datasets", "Datasets", "Datasets", "Datasets", "Datasets", "Datasets"],
        "Name": [
            "AI Fairness 360 (AIF360)",
            "Fairlearn",
            "TensorFlow Fairness Indicators",
            "What-If Tool (WIT)",
            "FairTest",
            "FairML",
            "Aequitas",
            "TIMSS",
            "PISA",
            "NELS:88",
            "EdNet",
            "KDDCUP 2015",
            "OULAD",
            "ASSISTments",
            "Statlog (German Credit Data)"
        ],
        "Description": [
            "IBM's extensible open-source toolkit for fairness metrics and bias mitigation.",
            "Microsoft's open-source toolkit to assess and improve model fairness, integrates with scikit-learn.",
            "Tools on TensorFlow Extended (TFX) for computing fairness metrics and visualizations.",
            "Interactive visual tool to probe model behavior and compare performance across data slices.",
            "Framework for automatically testing machine learning models for implicit bias.",
            "Python package to audit models for fairness by quantifying feature contributions to predictions.",
            "Open-source bias audit toolkit from University of Chicago for assessing fairness across multiple dimensions.",
            "International comparative data on student achievement in mathematics and science.",
            "Assesses 15-year-old students' reading, mathematics, and science literacy.",
            "Longitudinal study tracking students from 8th grade through post-secondary experiences.",
            "Large-scale dataset for educational purposes, student interaction data with online learning platforms.",
            "Dataset from an educational data mining competition focused on predicting student dropout.",
            "Demographic, activity, and assessment data from students in higher education.",
            "Widely used dataset from an online tutoring platform for mathematics.",
            "Classic dataset for fairness concerns in credit scoring, with parallels in educational resource allocation."
        ]
    }
    df_tools_datasets = pd.DataFrame(tools_data)
    st.dataframe(df_tools_datasets, hide_index=True)


    st.markdown("""
    ---

    ## 7 Challenges and Future Directions

    While significant progress has been made in understanding and addressing fairness in AI, several challenges remain, and new directions for research and development are constantly emerging, particularly in the dynamic field of educational AI.

    ### 7.1 Key Challenges:

    *   **Personalization Versus Data Privacy:** Balancing the benefits of personalized learning experiences with the imperative to protect sensitive student data is a complex challenge. Highly granular data, while enabling personalization, can also increase privacy risks.
    *   **Data-Related Challenges:**
        *   **Lack of Representative Data:** Many educational datasets suffer from a lack of diversity or are biased towards certain demographics, leading to models that perform poorly or unfairly for underrepresented groups.
        *   **Data Collection Ethics:** Ensuring that data is collected ethically, with informed consent, and without exacerbating existing power imbalances, is crucial.
    *   **The Definition and Measurement of Fairness:** There is no single, universally accepted definition of fairness. Different fairness metrics can conflict with each other, and deciding which metric is most appropriate for a given educational context is a non-trivial ethical and technical decision.
    *   **Trade-offs Between Fairness and Accuracy:** Improving fairness often comes at the cost of some reduction in overall model accuracy. Navigating this trade-off requires careful consideration of the societal impact and the specific goals of the AIED system.
    *   **Digital Divide:** The unequal access to technology and internet connectivity among students can lead to AIED systems that benefit privileged groups disproportionately, exacerbating existing educational inequalities.
    *   **Teacher and Student Adoption:** For AIED systems to be effective and equitable, they need to be adopted and trusted by both teachers and students. This requires transparent, explainable, and user-friendly systems that address user concerns about fairness and utility.

    ### 7.2 Future Directions:

    *   **Human-in-the-Loop AI:** Developing systems where human educators and students play an active role in monitoring, correcting, and improving AI decisions can enhance fairness and build trust.
    *   **Explainable AI (XAI) for Fairness:** Creating AIED systems that can clearly explain their recommendations and decisions, especially regarding fairness implications, will be crucial for accountability and user understanding.
    *   **Contextual Fairness:** Moving beyond generic fairness definitions to context-specific fairness, considering the unique pedagogical goals and societal implications within different educational settings.
    *   **Intersectional Fairness:** Addressing biases that arise from the intersection of multiple sensitive attributes (e.g., gender and race combined), which can be more complex than addressing single-attribute biases.
    *   **Proactive Fairness Design:** Integrating fairness considerations from the very initial stages of AIED system design, rather than treating it as an afterthought or a "fix" to be applied later.
    *   **Policy and Governance:** Developing robust policies, regulations, and governance structures to ensure the ethical and fair deployment of AI in education at systemic levels.

    By actively engaging with these challenges and pursuing these future directions, the field of AI in education can strive to build systems that are not only intelligent and effective but also fundamentally fair and equitable for all learners.
    """)

    st.markdown("""
    ---

    ## References

    The concepts and discussions in this lab are informed by contemporary research in the field of fairness, accountability, and transparency in AI. Key references include:

    *   [32] ... (Placeholder for a specific reference like FairAIE or others mentioned in the original notebook that might have specific citations)
    *   Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on fairness in recommender systems. *ACM Computing Surveys (CSUR)*, 54(4), 1-35.
    *   Barocas, S., Hardt, M., & Narayanan, A. (2017). *Fairness and machine learning: Limitations and opportunities*. Online.
    *   Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction. *arXiv preprint arXiv:1703.00056*.
    *   Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In *Proceedings of the 3rd innovations in theoretical computer science conference* (pp. 214-223).
    *   Verma, S., & Rubin, J. (2018). Fairness definitions explained. In *Proceedings of the 2018 IEEE/ACM international workshop on software fairness* (pp. 1-7).
    *   A. Narayanan, Fairness and machine learning. (2020). *https://fairmlbook.org/*
    *   ... (Other relevant academic papers, books, or online resources)
    """)
