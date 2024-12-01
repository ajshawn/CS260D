# #!/usr/bin/env python3

# # Import necessary libraries
# import openai
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np
# import os
# from dotenv import load_dotenv
# from tqdm import tqdm
# import pandas as pd
# import json

# # Load environment variables
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set as an environment variable

# client = openai.OpenAI()

# # Prepare data
# group_coreset_sentences = json.load(open("/local2/shawn/SimTrans/outputs/coreset_data_epoch_5.json"))['0']

# group_different_gen_df = pd.read_csv("/local2/shawn/SimTrans/tsne/different_gen.csv")
# # Filter by label == 0
# group_different_gen_df = group_different_gen_df[group_different_gen_df['labels'] == 0]
# group_different_gen_sentences = group_different_gen_df['text'].tolist()

# group_similar_gen_df = pd.read_csv("/local2/shawn/SimTrans/tsne/similar_gen.csv")
# # Filter by label == 0
# group_similar_gen_df = group_similar_gen_df[group_similar_gen_df['labels'] == 0]
# group_similar_gen_sentences = group_similar_gen_df['text'].tolist()

# group_dev_df = pd.read_csv("/local2/shawn/SimTrans/data/emotion/dev.csv")
# # Filter by label == 0
# group_dev_df = group_dev_df[group_dev_df['labels'] == 0]
# group_dev_sentences = group_dev_df['text'].tolist()

# label_names = ['Coreset', 'Different Gen', 'Similar Gen', 'Dev']

# # Combine sentences and labels
# sentences = group_coreset_sentences + group_different_gen_sentences + group_similar_gen_sentences + group_dev_sentences
# labels = []
# for label_name, group in zip(label_names, [group_coreset_sentences, group_different_gen_sentences, group_similar_gen_sentences, group_dev_sentences]):
#     labels.extend([label_name] * len(group))
    
# # Function to get embeddings from OpenAI API
# def get_embeddings(sentences, model="text-embedding-3-large"):
#     print("Embedding sentences...")
#     embeddings = []
#     for sentence in tqdm(sentences):
#         response = client.embeddings.create(
#             input=sentence,
#             model=model
#         )
#         embedding = response.data[0].embedding
#         embeddings.append(embedding)
#     return np.array(embeddings)

# # Embed sentences
# embeddings = get_embeddings(sentences)

# # Dimensionality reduction
# print("Reducing dimensionality...")
# reducer = TSNE(n_components=2, random_state=42, perplexity=5)
# embeddings_2d = reducer.fit_transform(embeddings)

# # Plotting
# print("Plotting embeddings...")
# plt.figure(figsize=(10, 10))
# colors = {
#     'Coreset': 'blue',
#     'Different Gen': 'red',
#     'Similar Gen': 'green',
#     'Dev': 'purple'
# }
# for label in set(labels):
#     idx = [i for i, l in enumerate(labels) if l == label]
#     plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[label])
# plt.legend()
# plt.title('Sentence Embeddings Visualization using OpenAI API')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.savefig('sentence_embeddings.png')

#!/usr/bin/env python3

# Import necessary libraries
import openai
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import json
from scipy.spatial import ConvexHull  # Added for convex hull computation

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set as an environment variable

for label_num in [0,1,2,3,4,5]:
    # Prepare data
    # group_coreset_sentences = json.load(open("/local2/shawn/SimTrans/outputs/coreset_data_epoch_5.json"))[str(label_num)]

    # group_different_gen_df = pd.read_csv("/local2/shawn/SimTrans/tsne/different_gen.csv")
    # group_different_gen_df = group_different_gen_df[group_different_gen_df['labels'] == label_num]
    # group_different_gen_sentences = group_different_gen_df['text'].tolist()

    # group_similar_gen_df = pd.read_csv("/local2/shawn/SimTrans/tsne/similar_gen.csv")
    # group_similar_gen_df = group_similar_gen_df[group_similar_gen_df['labels'] == label_num]
    # group_similar_gen_sentences = group_similar_gen_df['text'].tolist()

    # group_dev_df = pd.read_csv("/local2/shawn/SimTrans/data/emotion/dev.csv")
    # group_dev_df = group_dev_df[group_dev_df['labels'] == label_num]
    # group_dev_sentences = group_dev_df['text'].tolist()

    # label_names = ['Coreset', 'Different Gen', 'Similar Gen', 'Dev']

    # # Combine sentences and labels
    # sentences = group_coreset_sentences + group_different_gen_sentences + group_similar_gen_sentences + group_dev_sentences
    # labels = []
    # for label_name, group in zip(label_names, [group_coreset_sentences, group_different_gen_sentences, group_similar_gen_sentences, group_dev_sentences]):
    #     labels.extend([label_name] * len(group))

    group_shots_df = pd.read_csv(f"/local2/shawn/SimTrans/data/emotion/20shot.csv")
    group_shots_df = group_shots_df[group_shots_df['labels'] == label_num]
    group_shots_sentences = group_shots_df['text'].tolist()

    group_dev_df = pd.read_csv("/local2/shawn/SimTrans/data/emotion/dev.csv")
    group_dev_df = group_dev_df[group_dev_df['labels'] == label_num]
    group_dev_sentences = group_dev_df['text'].tolist()

    group_zero_shot_gen_df = pd.read_csv(f"/local2/shawn/SimTrans/data/emotion/llama3-8b-gen_6000.csv")
    group_zero_shot_gen_df = group_zero_shot_gen_df[group_zero_shot_gen_df['labels'] == label_num]
    group_zero_shot_gen_sentences = group_zero_shot_gen_df['text'].tolist()

    group_few_shot_gen_df = pd.read_csv(f"/local2/shawn/SimTrans/data/emotion/llama3-8b-20shot-6000.csv")
    group_few_shot_gen_df = group_few_shot_gen_df[group_few_shot_gen_df['labels'] == label_num]
    group_few_shot_gen_sentences = group_few_shot_gen_df['text'].tolist()

    label_names = ['20 Shots', 'Dev', 'Zero Shot Gen', 'Few Shot Gen']

    # Combine sentences and labels
    sentences = group_shots_sentences + group_dev_sentences + group_zero_shot_gen_sentences + group_few_shot_gen_sentences
    labels = []
    for label_name, group in zip(label_names, [group_shots_sentences, group_dev_sentences, group_zero_shot_gen_sentences, group_few_shot_gen_sentences]):
        labels.extend([label_name] * len(group))

    # Function to get embeddings from OpenAI API
    client = openai.OpenAI()

    def get_embeddings(sentences, model="text-embedding-3-large"):
        print("Embedding sentences...")
        embeddings = []
        for sentence in tqdm(sentences):
            response = client.embeddings.create(
                input=sentence,
                model=model
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return np.array(embeddings)

    # Embed sentences
    embeddings = get_embeddings(sentences)

    # # Save embeddings to a file
    # np.save('embeddings.npy', embeddings)
    # print("Embeddings saved to 'embeddings.npy'.")

    # Dimensionality reduction
    print("Reducing dimensionality...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plotting
    print("Plotting embeddings...")
    plt.figure(figsize=(10, 10))
    colors = {
        '20 Shots': 'blue',
        'Dev': 'red',
        'Zero Shot Gen': 'green',
        'Few Shot Gen': 'purple'
    }

    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[label])

        # Compute and plot convex hull
        points = embeddings_2d[idx]
        if len(points) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            # Close the hull by appending the first point at the end
            hull_points = np.concatenate([hull_points, hull_points[:1]])
            plt.plot(hull_points[:, 0], hull_points[:, 1], c=colors[label], linestyle='--')

    plt.legend()
    plt.title(f'Sentence Embeddings Visualization with label={label_num} for llama3-8b')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f'sentence_embeddings_labels={label_num}_llama.png')
    print("Plot saved to 'sentence_embeddings.png'.")
