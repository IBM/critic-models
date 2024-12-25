import json
import os.path
from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class InferEmbeddings(InferenceBase):

    def __init__(self, model, base_model_for_lora, data_path):
        super().__init__(model, base_model_for_lora, data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        new_data = {}
        for conversation in self.data:
            last_answer = self.data[conversation][-1]["content"]
            new_data[conversation] = last_answer
        self.data = new_data


    def load_model(self):
        load_args = {}
        if 'phi' in self.model_name.lower():
            load_args = {'trust_remote_code': True}
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, 
                                          device_map="auto", **load_args)
        return model


    def embed_inputs(self, tasks, batch_size=8):
        if len(tasks) == 0:
            print("!!!!!!")
        if len(tasks) < batch_size:
            all_embeddings = self._embed(tasks)
        else:
            print("RUNNING EMBEDDINGS...")
            all_embeddings = []
            for i in tqdm(range(0, len(tasks), batch_size)):
                all_embeddings.extend(self._embed(tasks[i:i+batch_size]))
        all_embeddings = torch.vstack(all_embeddings)
        return all_embeddings
    
    def _embed(self, batch):
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                truncation=True, max_length=2048).to('cuda:0')
        with torch.no_grad():
            embeddings = self.inference_model(**inputs).last_hidden_state
        actual_tokens_lengths = torch.sum(inputs['attention_mask'], axis=1)
        embeddings_mean = []
        for i, embedding in enumerate(embeddings):
            embedding_to_calc_mean = embedding[:actual_tokens_lengths[i]]
            mean_embedding = torch.mean(embedding_to_calc_mean, axis=0).cpu()
            embeddings_mean.append(mean_embedding)
        return embeddings_mean


class EncoderOnly(InferEmbeddings):
    def load_model(self):
        model = SentenceTransformer(self.model_name, trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        return model

    def embed_inputs(self, tasks, batch_size=8):
        torch.cuda.empty_cache()
        embeddings = self.inference_model.encode(tasks, batch_size=batch_size, prompt="", normalize_embeddings=True)
        return torch.tensor(embeddings, dtype=torch.float16)


def main(path_to_responses, embedding_model, path_to_decomposition, out_dir, encoder_only):
    with open(path_to_decomposition, 'rt') as f:
        decomposition = json.load(f)
    decomposition = {t.strip(): decomposition[t] for t in decomposition}
    for task in decomposition:
        if len(decomposition[task]) == 0:
            decomposition[task] = [task]

    model_class = InferEmbeddings if not encoder_only else EncoderOnly
    inference_model = model_class(model=embedding_model, base_model_for_lora=None, data_path=path_to_responses)
    data = inference_model.data
    sorted_task_list = get_sorted_task_list(data)
    path = os.path.join(out_dir, "sorted_tasks.json")
    with open(path, 'wt') as f:
        str_dump = json.dumps(sorted_task_list, indent=2)
        f.write(str_dump)

    all_atomics = set()
    for task in decomposition:
        all_atomics = all_atomics.union(decomposition[task])
    all_atomics_sorted = sorted(list(all_atomics))
    path = os.path.join(out_dir, "sorted_atomics.json")
    with open(path, 'wt') as f:
        str_dump = json.dumps(all_atomics_sorted, indent=2)
        f.write(str_dump)

    path = os.path.join(out_dir, "atomics.npy")
    if not os.path.exists(path):
        semantic_embedding_atomics = inference_model.embed_inputs(all_atomics_sorted)
        np.save(path, semantic_embedding_atomics)
    semantic_embedding_atomics = np.load(path)

    path = os.path.join(out_dir, "tasks.npy")
    if not os.path.exists(path):
        semantic_embedding_tasks = inference_model.embed_inputs(sorted_task_list)
        np.save(path, semantic_embedding_tasks)
    semantic_embedding_tasks = np.load(path)

    sorted_outputs_list = [data[task] for task in sorted_task_list]
    path = os.path.join(out_dir, "outputs.npy")
    if not os.path.exists(path):
        semantic_embedding_outputs = inference_model.embed_inputs(sorted_outputs_list)
        np.save(path, semantic_embedding_outputs)
    semantic_embedding_outputs = np.load(path)

    distance_input_output_embeddings = semantic_embedding_tasks - semantic_embedding_outputs
    path = os.path.join(out_dir, "distance_between_input_output.npy")
    np.save(path, distance_input_output_embeddings)


    per_task_atomic_embeddings = []
    for task in sorted_task_list:
        atomics_for_task = decomposition[task]
        atomics_embedding_current_task = []
        for atom in atomics_for_task:
            if atom in all_atomics_sorted:
                index = all_atomics_sorted.index(atom)
                atomics_embedding_current_task.append(semantic_embedding_atomics[index])
            else:  # atom is a combined instruction (a case where decomposition failed)
                index_task = sorted_task_list.index(atom)
                atomics_embedding_current_task.append(semantic_embedding_tasks[index_task])
        per_task_atomic_embeddings.append(atomics_embedding_current_task)


    path_mean = os.path.join(out_dir, "task_as_mean_atomics.npy")
    if not os.path.exists(path_mean):
        mean_per_task_embedding_atomics = []
        for i, task in enumerate(sorted_task_list):
            atomics_embeds = np.array(per_task_atomic_embeddings[i])
            mean_atomic_for_task = np.mean(atomics_embeds, axis=0)
            mean_per_task_embedding_atomics.append(mean_atomic_for_task)
            assert mean_atomic_for_task.shape == atomics_embeds[0].shape, f"{mean_atomic_for_task.shape} {atomics_embeds.shape}"
        np.save(path_mean, mean_per_task_embedding_atomics)



    path = os.path.join(out_dir, "input_and_output.npy")
    if not os.path.exists(path):
        sorted_concatenated = [f"{task} {data[task]}" for task in sorted_task_list]
        semantic_embedding_inp_and_out = inference_model.embed_inputs(sorted_concatenated, batch_size=8)
        np.save(path, semantic_embedding_inp_and_out)




def get_sorted_task_list(data):
    tasks = list(data.keys())
    tasks = [t.strip() for t in tasks]
    tasks = sorted(tasks)
    return tasks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--initial_responses")
    parser.add_argument("--embedding_model")
    parser.add_argument("--decomposition")
    parser.add_argument("--out_dir")
    parser.add_argument("--encoder_only", default=False, action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.cuda.empty_cache()
    main(path_to_responses=args.initial_responses, embedding_model=args.embedding_model,
         path_to_decomposition=args.decomposition, out_dir=args.out_dir, encoder_only=args.encoder_only)