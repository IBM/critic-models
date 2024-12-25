
import pandas as pd

def prepare_df(pairs, out_path):
    all_samples = []
    for generator_model in pairs:
        for critic_model in pairs[generator_model]:
            pair = pairs[generator_model][critic_model]
            improved_relative = pair.compare_responses('relative_total')
            improved_total_diff = pair.compare_responses('sum')
            improved_mean_diff = pair.compare_responses('mean')
            for task in pair.get_all_tasks():
                scores = pair.get_scores_for_task(task)
                sum_initial = sum(scores['initial'].values())
                sum_revised = sum(scores['revised'].values())
                num_constraints = len(scores['initial'])
                sample = {
                    "task": task,
                    "num_constraints": num_constraints,
                    "generator_model": generator_model,
                    "initial_score_sum": sum_initial,
                    "critic_model": critic_model,
                    "revised_score_sum": sum_revised,
                    "improved_relative": improved_relative[task],
                    "improved_sum": improved_total_diff[task],
                    "improved_mean": improved_mean_diff[task]
                }
                all_samples.append(sample)
    df = pd.DataFrame(all_samples)
    df.to_csv(out_path)
    return df
