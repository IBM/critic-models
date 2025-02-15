import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


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


def generate_color_map():

    # Define color maps for different model families
    color_maps = {
        "deepseek": cm.Reds,  # Red gradient
        "mistral": cm.Blues,  # Blue gradient
        "gemma": cm.Greens,  # Green gradient
        "llama": cm.Purples,  # Purple gradient
        "qwen": cm.Oranges  # Orange gradient
    }

    # Define models and their relative parameter sizes (larger = darker)
    models = {
        "deepseek": ["Deepseek-v3"],  # Single model
        "mistral": ["Mistral-large"],
        "gemma": ["Gemma-2-2b", "Gemma-2-9b"],
        "llama": ["Llama-3.2-1b", "Llama-3.2-3b", "Llama-3.1-8b", "Llama3.3-70b", "Llama3.1-405b"],
        "qwen": ["Qwen2.5-0.5b", "Qwen2.5-1.5b", "Qwen2.5-3b", "Qwen2.5-7b", "Qwen2.5-72b"]
    }

    # Normalize parameter sizes for gradient assignment
    parameter_sizes = {
        "deepseek": [1],  # Single model, so single shade
        "mistral": [1],  # Single model
        "gemma": [1, 2],
        "llama": [3, 6, 9, 12, 15],
        "qwen": [3, 6, 9, 12, 15]
    }

    # Normalize parameter values between 0 (light) and 1 (dark)
    normalized_sizes = {k: np.interp(v, (min(v), max(v)), (0.3, 0.8)) for k, v in parameter_sizes.items()}

    # Generate dictionary mapping models to colors
    model_colors = {
        model: mcolors.to_hex(color_maps[family](normalized_sizes[family][i]))
        for family, models_list in models.items()
        for i, model in enumerate(models_list)
    }
    return model_colors

