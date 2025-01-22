import numpy as np
import json

small_model_0_shot_path = "/Users/gililior/research/datasets/arena_data_pivot_jan25/decomposition-evaluation-llama3.1-70b.arena_inference_gemma_2b_train.json"
big_model_0_shot_path = "/Users/gililior/research/datasets/arena_data_final/llm_aaj_constraints_evaluation/initial/decomposition-evaluation-llama3.1-70b.llama3-8b-init-response-train.json"
big_model_refine_small_one_step_path = "/Users/gililior/research/datasets/arena_data_pivot_jan25/decomposition-evaluation-llama3.1-70b.llama3_8b_refine_gemma_2b_one_step_full.json"
big_model_refine_small_two_steps_path = "/Users/gililior/research/datasets/arena_data_pivot_jan25/decomposition-evaluation-llama3.1-70b.llama3_8b_refine_gemma_2b_two_steps_full.json"
big_model_self_critic_path = "/Users/gililior/research/datasets/arena_data_final/llm_aaj_constraints_evaluation/revised/decomposition-evaluation-llama3.1-70b.init-llama3-8b-train.critic-llama3-8b.revise-llama3-8b.json"
all_paths = {
    "small_model_0_shot": small_model_0_shot_path,
    "big_model_0_shot": big_model_0_shot_path,
    "big_model_refine_small_one_step": big_model_refine_small_one_step_path,
    "big_model_refine_small_two_steps": big_model_refine_small_two_steps_path,
    "big_model_self_critic": big_model_self_critic_path
}



def calculate():
    all_jsons = {}
    all_tasks = set()
    for k in all_paths:
        with open(all_paths[k], 'r') as f:
            all_jsons[k] = json.load(f)
            all_tasks.update(all_jsons[k].keys())

    all_scores = {k: [] for k in all_jsons}
    for task in all_tasks:
        for k in all_jsons:
            scores = list(all_jsons[k][task]["scores"].values())
            if len(scores) == 0:
                continue
            scores = [0 if s == 'ERR' else s for s in scores]
            mean_score = np.mean(scores)
            all_scores[k].append(mean_score)

    for k in all_scores:
        print(f"{k}: {np.mean(all_scores[k])}")

if __name__ == '__main__':
    calculate()