import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from vectorize_sentences import vectorize, pickle_file


def check_file_exists(file_path):
    """ Check if the file exists """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File `{file_path}` not found")


def load_data(file_path):
    """ Load the DataFrame from a pickle file """

    df = pd.read_pickle(file_path)
    return df


def cosine_similarity(v1, v2):
    """ Compute the cosine similarity between two vectors """

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / (norm_v1 * norm_v2)

def jaccard_similarity(v1,v2):
    """ Compute the Jaccard similarity between two vectors """
    set1 = set(v1)
    set2 = set(v2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def manhattan_similarity(v1, v2):
    """ Compute the Manhattan similarity between two vectors """
    max_distance = sum(max(x, y) for x, y in zip(v1, v2))
    distance = sum(abs(x - y) for x, y in zip(v1, v2))
    return 1 - (distance / max_distance)



def compare_vectors(vectors, df):
    """ Compare vectors with the vectors from the dataframe and keep top X """

    similarities = []

    for idx, vector in enumerate(df['vectors']):
        similarity = cosine_similarity(vectors, vector)
        similarities.append(similarity)

    df['Similarity'] = similarities
    df = df.sort_values(by='Similarity', ascending=False)

    return df


def display_results(df, limit):
    """ Display the results in CLI """

    print(df.head(limit))


def plot_results(df, limit, file_path):
    """ Plot the results """

    # Sort the DataFrame by similarity and take the top 5 most similar vectors
    top_k_df = df.nlargest(limit, 'Similarity')

    # Create a scatter plot for the top 5 similarities
    plt.figure(figsize=(12, 8))  # Increase the figure size for more room
    plt.scatter(top_k_df['Text'].str[:30],
                top_k_df['Similarity'], color='blue', s=100)

    # Add titles and labels
    plt.title(
        'Top Cosine Similarities between Input Vector and Reference', fontsize=16)
    plt.xlabel('References', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)

    # Rotate x labels and add padding
    # Rotate and increase font size
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()  # Adjust layout to make room for the labels

    # Save the plot to a file
    plt.savefig(file_path)


def main(text, limit, plot_file_path):
    vectors = vectorize(text)

    # Load the DataFrame
    df = load_data(pickle_file)

    # Compare vectors
    df = compare_vectors(vectors, df)

    # Display results
    display_results(df, limit)

    # Plot results
    if plot_file_path:
        plot_results(df, limit, plot_file_path)


if __name__ == "__main__":
    # Get text as input and n limit defaults to 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str,
                        required=True, help="Input text to compare")
    parser.add_argument("--limit", "-l", type=int, default=1,
                        help="Number of similarities to display")
    parser.add_argument("--plot_file", "-f", type=str,
                        help="File path to save the plot")
    args = parser.parse_args()

    text = args.text
    limit = args.limit
    plot_file_path = args.plot_file

    # Check Pickle file exists
    check_file_exists(pickle_file)

    main(text, limit, plot_file_path)
