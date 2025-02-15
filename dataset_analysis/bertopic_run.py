import numpy as np
from bertopic import BERTopic
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():

    # cluster_model = KMeans(n_clusters=12)
    # dim_model = PCA(n_components=50)
    tasks = load_dataset("gililior/wild-if-eval", split="test")["task"]
    sentence_model = SentenceTransformer("all-mpnet-base-v2", device='mps')

    topic_model = BERTopic(embedding_model=sentence_model)
    topics, probs = topic_model.fit_transform(tasks)
    topics = np.array(topics)
    only_high_certainty = topics[probs > 0.8]
    from collections import Counter
    c = Counter(only_high_certainty)
    for topic, count in sorted(c.items()):
        print(f"Topic {topic}: {count} tasks")
        print(topic_model.get_topic(topic))
        print("========================================")
    print(*topic_model.get_topics(), sep='\n\n')

if __name__ == '__main__':
    main()
