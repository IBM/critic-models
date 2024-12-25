import re

def process_response(response):
    regex_pattern = r"\[*START\]*(.*?)\[*END\]*"
    flags = re.DOTALL
    summary = re.findall(regex_pattern, response, flags=flags)
    if len(summary) == 0:
        regex_pattern = r"\[*START\]*(.*)"
        summary = re.findall(regex_pattern, response, flags=flags)
    if len(summary) == 0:
        regex_pattern = r"(.*?)\[*END\]*"
        summary = re.findall(regex_pattern, response, flags=flags)
    if len(summary) == 0:
        regex_pattern = r"<(.*)>"
        summary = re.findall(regex_pattern, response, flags=flags)
    if len(summary) == 0:
        regex_pattern = r"<(.*)"
        summary = re.findall(regex_pattern, response, flags=flags)
    if len(summary) == 0:
        summary = [response]
    only_summary = summary[-1]
    explanation = response.replace(only_summary, "")
    return only_summary.strip(), explanation
