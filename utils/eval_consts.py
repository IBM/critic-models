SYSTEM_MESSAGE_EVAL = """
You are a helpful assistant, trained to asses the quality of outputs for some task.
""".strip()

PROMPT_EVAL = """
Below is some task instruction and a corresponding response to evaluate.
Your job is to answer whether the specified constraint is satisfied or not.

###Task instruction:
{instruction}

###Response:
{response}

###Question:
Is the following constraint satisfied: "{constraint}"?

###Answer:
""".strip()

LLAMA3_CHAT_FORMAT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}
<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_msg}
<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>
""".strip()
