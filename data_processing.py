import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from gensim.models import KeyedVectors
import torch

# Load the spaCy language model once, globally
nlp = spacy.load("en_core_web_sm")

def tokenize_lyrics_spacy(lyrics_str):
    """
    Tokenize lyrics using spaCy, keeping only alphabetic words.
    """
    doc = nlp(lyrics_str)
    valid_words = [token.text.lower() for token in doc if token.is_alpha]
    return valid_words

def tokenize_dataset(df):
    """
    Tokenize the 'words' column in the dataframe using the spaCy tokenizer.
    """
    df['tokenized_words'] = df['words'].apply(tokenize_lyrics_spacy)
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and validation sets.
    """
    df_train, df_valid = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Training set size: {len(df_train)} rows")
    print(f"Validation set size: {len(df_valid)} rows")
    return df_train, df_valid

def build_vocab(df_train):
    """
    Build a vocabulary from the tokenized training data.
    """
    all_train_tokens = []
    for tokens in df_train['tokenized_words']:
        all_train_tokens.extend(tokens)

    unique_words = sorted(list(set(all_train_tokens)))
    word2idx = {w: i for i, w in enumerate(unique_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    
    print(f"Vocabulary size: {vocab_size}")
    return word2idx, idx2word, vocab_size

def load_word2vec(repo_id="NathaNn1111/word2vec-google-news-negative-300-bin", filename="GoogleNews-vectors-negative300.bin"):
    """
    Download and load the pretrained Word2Vec model from Hugging Face.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=True)
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(f"Word2Vec model loaded successfully with {len(w2v_model)} words.")
    return w2v_model

def create_dataloader(tokenized_data, word2idx, sequence_length=5, batch_size=32):
    """
    Convert tokenized_data (list of token lists) into sequences of indices 
    and create a PyTorch DataLoader.
    """
    x_data = []
    y_data = []

    for tokens in tokenized_data:
        # Convert tokens to indices
        indices = [word2idx[t] for t in tokens if t in word2idx]
        
        # Create sliding windows of size `sequence_length`
        for i in range(len(indices) - sequence_length):
            x_seq = indices[i:i+sequence_length]  # input
            y_seq = indices[i+sequence_length]    # next word (target)
            x_data.append(x_seq)
            y_data.append(y_seq)
    
    x_tensor = torch.tensor(x_data, dtype=torch.long)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def process_file(file_path):
    """
    Process a CSV-like text file and return a DataFrame with 
    columns: ['singer_name', 'song_name', 'words'].
    """
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            parts = line.strip().split(',')
            singer_name = parts[0].strip()
            song_name = parts[1].strip()
            words = ','.join(parts[2:]).strip()
            data.append([singer_name, song_name, words])
        return pd.DataFrame(data, columns=['singer_name', 'song_name', 'words'])

def data_process(df_train, sequence_length=5, batch_size=32, embedding_dim=300):
    """
    Full data processing pipeline: 
    1. Tokenize 
    2. Split
    3. Build vocab
    4. Load Word2Vec & create embedding matrix
    5. Create DataLoaders
    """
    # Step 1: Tokenize
    df_tokenized = tokenize_dataset(df_train)

    # Step 2: Split data
    df_train_split, df_valid_split = split_data(df_tokenized)

    # Step 3: Build vocab
    word2idx, idx2word, vocab_size = build_vocab(df_train_split)

    # Step 4: Load Word2Vec and create custom embedding matrix
    w2v_model = load_word2vec()
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]

    # Step 5: Create DataLoaders
    train_loader = create_dataloader(df_train_split['tokenized_words'], word2idx, sequence_length, batch_size)
    val_loader = create_dataloader(df_valid_split['tokenized_words'], word2idx, sequence_length, batch_size)

    return train_loader, val_loader, word2idx, idx2word, vocab_size, embedding_matrix
