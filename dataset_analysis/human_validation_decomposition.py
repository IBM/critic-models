import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import json
import os

# Load JSON file with task decompositions
st.sidebar.title("Upload JSON File")
uploaded_file = st.sidebar.file_uploader("Upload a JSON file", type=["json"])

tasks = []
if uploaded_file is not None:
    tasks_dict = json.load(uploaded_file)
    tasks = list(tasks_dict.items())  # Convert dict to list of (task, decomposition) tuples

# Title
st.title("Task Decomposition Evaluation")

# Initialize session state for task index and results storage
if "task_index" not in st.session_state:
    st.session_state.task_index = 0
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = []

name = uploaded_file.split("-")[-1].replace(".json", "")
# Load existing evaluation results if available
csv_filename = os.path.join(f"evaluation_results_{name}.csv")
if os.path.exists(csv_filename):
    existing_results = pd.read_csv(csv_filename)
    evaluated_tasks = set(existing_results["Task"].tolist())
    st.session_state.evaluation_results = existing_results.to_dict(orient='records')
else:
    evaluated_tasks = set()

# Skip already evaluated tasks
tasks = [task for task in tasks if task[0] not in evaluated_tasks]

# Iterate through tasks
if tasks and st.session_state.task_index < len(tasks):
    task, decomposition = tasks[st.session_state.task_index]
    decomposition_text = "\n".join(decomposition)

    st.markdown("**Task:**")
    st.markdown(task)
    st.markdown("**Decomposition:**")
    st.markdown("<ul>" + "".join([f'<li>{step}</li>' for step in decomposition]) + "</ul>", unsafe_allow_html=True)

    # Evaluation Criteria with Descriptions
    st.subheader("Evaluate the Decomposition")
    measure_definitions = {
        "Correctness": "Does the decomposition faithfully reflect the original task?",
        "Completeness": "Does the decomposition account for all essential constraints?",
        "Independence": "Are the constraints distinct and self-sufficient?"
    }

    ratings = {}
    cols = st.columns(len(measure_definitions))
    for i, (measure, definition) in enumerate(measure_definitions.items()):
        with cols[i]:
            st.markdown(f"**{measure}**: {definition}")
            ratings[measure] = st.slider(f"Rate {measure}", 1, 5, 3)

    # Submit button
    if st.button("Submit Evaluation"):
        ratings["Task"] = task
        st.session_state.evaluation_results.append(ratings)

        # Save to CSV
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        results_df.to_csv(csv_filename, index=False)

        st.write("### Evaluation Results")
        st.dataframe(results_df)

        # Move to next task
        st.session_state.task_index += 1
        st.rerun()
else:
    st.write("No more tasks to evaluate or no file uploaded.")

    # Provide a download button for results
    if st.session_state.evaluation_results:
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="evaluation_results_gili.csv",
            mime="text/csv"
        )