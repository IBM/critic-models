import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import json
import os

# Load JSON file with task decompositions
st.sidebar.title("Upload JSON File")
uploaded_file = st.sidebar.file_uploader("Upload a csv file", type=["csv"])

tasks_df = None
if uploaded_file is not None:
    tasks_df = pd.read_csv(uploaded_file)

# Title
st.title("LLM as a Judge Evaluation")

# Initialize session state for task index and results storage
if "task_index" not in st.session_state:
    st.session_state.task_index = 0
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = pd.DataFrame(columns=["task", "constraint", "response", "model", "binary_score", "human_eval"])

# Load existing evaluation results if available
csv_filename = os.path.join(f"llm_aaj_annotations.csv")
if os.path.exists(csv_filename):
    existing_results = pd.read_csv(csv_filename)
    st.session_state.task_index = len(existing_results)
    st.session_state.evaluation_results = existing_results
else:
    evaluated_tasks = set()


# Iterate through tasks
if tasks_df is not None and st.session_state.task_index < len(tasks_df):
    row = tasks_df.iloc[st.session_state.task_index]

    st.markdown("**Task:**")
    st.markdown(row["orig_task"])
    st.markdown("**Constraint:**")
    st.markdown(row["constraint"])
    st.markdown("**Response:**")
    st.markdown(row["response"])

    # binary answer yes or no
    st.markdown(f"Is the constraint \"{row['constraint']}\" satisfied?")
    st.radio("Answer", ["Yes", "No"], key="prediction")

    # save the answer to the csv
    if st.button("Save Answer"):
        new_row = {
            "task": row["orig_task"],
            "constraint": row["constraint"],
            "response": row["response"],
            "model": row["model"],
            "binary_score": row["binary_score"],
            "human_eval": 1 if st.session_state.prediction == "Yes" else 0
        }
        st.session_state.evaluation_results = pd.concat([st.session_state.evaluation_results, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.evaluation_results.to_csv(csv_filename, index=False)
        st.session_state.task_index += 1
        st.rerun()
else:
    st.write("No more tasks to evaluate or no file uploaded.")