import json

from prepare_data.classify_constrained_generation_tasks import ConstrainedGenerationClassificationWMV, BaseDataset, \
    ConstrainedGenerationClassificationRITS
from utils.filter_data_consts import DECOMPOSE_PROMPT
import re
from argparse import ArgumentParser


def filter_answer(text):
    if "Translated Constraints:" not in text:
        return []
    index = text.find("Translated Constraints:") + len("Translated Constraints:")
    answer = text[index:].strip()
    list_items = re.split(r'(?=\n\d+\.\s*[A-Z])', answer)

    # Remove empty strings from the list
    list_items = [item.strip() for item in list_items if item]

    # Remove numbers from the list items
    list_items = [re.sub(r'^\d+\.', '', item).strip() for item in list_items]
    if "" in list_items:
        print(answer)
    return list_items

class DecomposerWMV(ConstrainedGenerationClassificationWMV):
    def _infer(self, task):
        message = DECOMPOSE_PROMPT.format(instruction=task)
        answer = self.get_answer(message)
        generated_text = answer["results"][0]["generated_text"]
        processed_answer = filter_answer(generated_text)
        return processed_answer

    def get_name(self):
        return "decomposition"


class DecomposerRITS(ConstrainedGenerationClassificationRITS):
    def _infer(self, task):
        message = DECOMPOSE_PROMPT.format(instruction=task)
        answer = self.get_answer(message)
        generated_text = answer["results"][0]["generated_text"]
        processed_answer = filter_answer(generated_text)
        return processed_answer

    def get_name(self):
        return "decomposition"


class ArenaClassifiedData(BaseDataset):
    def __init__(self, name_or_path):
        super().__init__(name_or_path)

    def load_data(self, name_or_path):
        with open(name_or_path, 'rt') as f:
            data = json.load(f)
        return data

    def get_tasks_list(self):
        return list(self.data.keys())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--model_name")
    parser.add_argument("--platform", choices=['wmv', 'rits'], required=True)
    args = parser.parse_args()
    dataset = ArenaClassifiedData(args.data)
    if args.platform == 'wmv':
        decomposer = DecomposerWMV(dataset, args.model_name, max_new_tokens=1000)
    else:
        decomposer = DecomposerRITS(dataset, args.model_name, max_new_tokens=1000)
    decomposer.infer(args.out)