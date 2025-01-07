from argparse import ArgumentParser
from routing.self_critic_classifier import SelfCriticClassifier
from routing.multi_class_classifier import MulriCriticClassifier


def get_classifier_class(classifier_name):
    if classifier_name == 'self-critic':
        return SelfCriticClassifier
    elif classifier_name == 'multi-class':
        return MulriCriticClassifier
    else:
        raise ValueError(f'Invalid classifier name: {classifier_name}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier_name', type=str, choices=['self-critic', 'multi-class'], required=True,
                        help='Name of the classifier to use')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--df', type=str, required=True, help='Path to dataframe with labels')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA parameter r (rank)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA parameter alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA parameter dropout rate')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to results file for saving metrics and hyperparameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    classifier_cls = get_classifier_class(args.classifier_name)
    classifier = classifier_cls(args.config, args.df, args.model_name, args.learning_rate, args.batch_size,
                                args.num_epochs, args.weight_decay, args.lora_r, args.lora_alpha, args.lora_dropout,
                                args.results_file, args.seed)
    classifier.run_all()
