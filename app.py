
import streamlit as st
st.set_page_config(page_title="FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: FairAIED")
st.markdown("""
# FairAIED: Navigating Fairness, Bias, and Ethics in Educational AI Applications

Welcome to the **FairAIED** lab, an interactive environment designed to explore the critical concepts of fairness, bias, and ethics in the context of educational AI applications, particularly course recommendation systems.

Recommendation systems are pervasive, influencing decisions from entertainment to commerce. In education, they offer immense potential to guide students toward courses aligning with their interests, goals, and strengths. However, these systems, being data-driven, are susceptible to biases present in their training data. If unchecked, such biases can perpetuate or even amplify existing inequalities, leading to unfair or discriminatory recommendations for certain student groups.

This lab aims to provide a hands-on experience in understanding and addressing these challenges. Through interactive simulations and visualizations, you will:

*   **Understand Recommendation Systems**: Learn how these systems function and how they can inadvertently perpetuate existing biases.
*   **Explore Fairness Metrics**: Familiarize yourself with key fairness metrics such as Demographic Parity, Equal Opportunity, Equalized Odds, and Predictive Parity, and grasp their implications in educational contexts.
*   **Investigate Fairness Techniques**: Explore conceptual techniques to incorporate fairness constraints into recommendation algorithms.
*   **Analyze Trade-offs**: Understand the inherent trade-offs between fairness, accuracy, and diversity in recommendation systems.

Our journey will be informed by contemporary research in AI fairness, ensuring a robust and relevant learning experience. Use the navigation panel on the left to explore different aspects of fairness in educational AI.
""")
st.divider()

# Sidebar for global controls - placed here to be always visible
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

# Your code starts here - Navigation after global controls
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
# Your code ends
