import streamlit as st
import json
import random
import os

# Load the tasks from the uploaded JSON file
def load_tasks(uploaded_file, username):
    data = json.load(uploaded_file)

    key = f"{username}_annotation"
    tasks = []
    for item in data.values():
        if key not in item:
            tasks.append(item)

    return tasks, data

# Save the annotations
def save_annotation(output_path, username, conversation_id, choice, full_data):
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {}

    key = f"{username}_annotation"
    if key not in annotations:
        annotations[key] = {}

    annotations[key][conversation_id] = choice

    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    # Also update the original dataset
    if conversation_id in full_data:
        full_data[conversation_id][key] = choice

# Download annotations
def download_annotations(full_data):
    st.download_button(
        label="Download Current JSON",
        data=json.dumps(full_data, indent=2),
        file_name="updated_dataset.json",
        mime="application/json"
    )

# Main app
st.set_page_config(page_title="Constraint Voting Platform", layout="wide")

# Initialize session state
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'task_idx' not in st.session_state:
    st.session_state.task_idx = 0
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'full_data' not in st.session_state:
    st.session_state.full_data = {}
if 'shuffle_order' not in st.session_state:
    st.session_state.shuffle_order = True
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Upload file
if not st.session_state.file_uploaded:
    st.title("Upload your JSON file")
    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_uploaded = True
        st.rerun()

# Page 1: Get Username and Show Instructions
elif st.session_state.username == "":
    st.title("Welcome to the Annotation Platform!")

    username = st.text_input("Please enter your name:")

    st.markdown("""
    ## Instructions
    - You will see two lists of constraints, side-by-side.
    - Your task is to **choose** which constraint list you think is better.
    - The order (left/right) will **shuffle randomly** each time.
    - Click the button under your preferred list or choose tie if you think they are equally good.
    """)

    if username:
        st.session_state.username = username
        st.session_state.tasks, st.session_state.full_data = load_tasks(st.session_state.uploaded_file, username)
        st.rerun()

else:
    # Page 2: Annotation Interface
    tasks = st.session_state.tasks
    idx = st.session_state.task_idx

    download_annotations(st.session_state.full_data)

    if idx >= len(tasks):
        st.success("You have completed all the tasks. Thank you!")
    else:
        task = tasks[idx]

        # Randomly shuffle left/right
        if st.session_state.shuffle_order:
            if random.random() < 0.5:
                left_label, right_label = "gpt", "llama"
                left_constraints = task['gpt4_constraints']
                right_constraints = task['llama3.1-8b_constraints']
            else:
                left_label, right_label = "llama", "gpt"
                left_constraints = task['llama3.1-8b_constraints']
                right_constraints = task['gpt4_constraints']
            st.session_state.left_label = left_label
            st.session_state.right_label = right_label
            st.session_state.left_constraints = left_constraints
            st.session_state.right_constraints = right_constraints
            st.session_state.shuffle_order = False

        st.title(f"Task {idx+1}/{len(tasks)}")
        st.subheader("Task Description:")
        st.write(task['task'])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Option 1")
            for c in st.session_state.left_constraints:
                st.markdown(f"- {c}")

            if st.button("Vote for Option 1", key=f"vote_left_{idx}"):
                save_annotation("annotations.json", st.session_state.username, task['conversation_id'], st.session_state.left_label, st.session_state.full_data)
                st.session_state.task_idx += 1
                st.session_state.shuffle_order = True
                st.rerun()

        with col2:
            st.markdown("### Option 2")
            for c in st.session_state.right_constraints:
                st.markdown(f"- {c}")

            if st.button("Vote for Option 2", key=f"vote_right_{idx}"):
                save_annotation("annotations.json", st.session_state.username, task['conversation_id'], st.session_state.right_label, st.session_state.full_data)
                st.session_state.task_idx += 1
                st.session_state.shuffle_order = True
                st.rerun()

        st.markdown("---")
        if st.button("Vote Tie", key=f"vote_tie_{idx}"):
            save_annotation("annotations.json", st.session_state.username, task['conversation_id'], "tie", st.session_state.full_data)
            st.session_state.task_idx += 1
            st.session_state.shuffle_order = True
            st.rerun()
