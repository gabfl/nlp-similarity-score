from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

# Load pre-trained model and tokenizer
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

source_file = "data/random_sentences.csv"
pickle_file = "data/df.pkl"


def read_file(file_path):
    """ Read csv file and return a pandas DataFrame """

    df = pd.read_csv(file_path)
    return df


def vectorize(text):
    """ Vectorize input text using pre-trained model """

    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = torch.mean(
        outputs.last_hidden_state, dim=1).detach().numpy()

    return embeddings[0]


def vectorize_text(df):
    """ Vectorize input text using pre-trained model """

    vectors = []  # Initialize an empty list to store the vectors

    for text in df['Text']:
        embeddings = vectorize(text)
        vectors.append(embeddings)  # Store the vector for this sentence

    df['vectors'] = vectors  # Assign the list of vectors to the DataFrame column

    return df


def pickle_data(df, file_path):
    """ Save the DataFrame to a pickle file """

    df.to_pickle(file_path)
    print("")
    print(f"DataFrame saved to `%s`" % file_path)


def main(source_file):

    df = read_file(source_file)
    df = vectorize_text(df)

    print(df.head())

    pickle_data(df, pickle_file)


if __name__ == "__main__":
    main(source_file)
