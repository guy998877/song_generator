import math
import torch
import torch.nn as nn
import torch.optim as optim

# Import from our other modules
from data_processing import data_process
from model import LyricsLSTMModel
from generate import generate_text

def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    """
    Train the LSTM model using provided train_loader.
    Prints loss and perplexity after each epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            # Initialize hidden state
            batch_size = inputs.size(0)
            state_h, state_c = model.init_hidden(batch_size)

            inputs, targets = inputs.to(device), targets.to(device)
            state_h, state_c = state_h.to(device), state_c.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, (state_h, state_c) = model(inputs, state_h, state_c)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        perplexity = math.exp(avg_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on a validation loader.
    Prints accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            batch_size = inputs.size(0)
            state_h, state_c = model.init_hidden(batch_size)

            inputs, targets = inputs.to(device), targets.to(device)
            state_h, state_c = state_h.to(device), state_c.to(device)

            outputs, (state_h, state_c) = model(inputs, state_h, state_c)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

def train_model_process(train_loader, vocab_size, embedding_dim, hidden_size, embedding_matrix, device, epochs=10, lr=1e-3):
    """
    Define, initialize, and train the model.
    """
    model = LyricsLSTMModel(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        dropout_p=0.5,
        padding_idx=0
    )
    # Copy pretrained embeddings
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    # Optionally, freeze embeddings:
    # model.embedding.weight.requires_grad = False

    train_model(model, train_loader, device, epochs, lr)
    return model

def evaluate_and_generate(model, val_loader, word2idx, idx2word, sequence_length, temperature, device):
    """
    Evaluate the model and generate text using the trained model.
    """
    # Evaluate
    evaluate_model(model, val_loader, device)

    # Example start tokens
    start_tokens = [word2idx[word] for word in ["goodbye", "norma", "jean"] if word in word2idx]
    if len(start_tokens) < sequence_length:
        start_tokens = ([0] * (sequence_length - len(start_tokens))) + start_tokens

    # Generate
    generated_lyrics = generate_text(
        model,
        start_tokens=start_tokens,
        word2idx=word2idx,
        idx2word=idx2word,
        max_length=20,
        temperature=temperature,
        device=device
    )
    print("\n=== Generated Lyrics ===")
    print(generated_lyrics)

def main(df_train, sequence_length=5, batch_size=32, embedding_dim=300, hidden_size=256, epochs=10, lr=1e-3, temperature=1.0):
    """
    Main function to orchestrate data processing, training, evaluation, and generation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Data Processing
    train_loader, val_loader, word2idx, idx2word, vocab_size, embedding_matrix = data_process(
        df_train, sequence_length, batch_size, embedding_dim
    )

    # Step 2: Train
    model = train_model_process(
        train_loader, vocab_size, embedding_dim, hidden_size, embedding_matrix, 
        device, epochs, lr
    )

    # Step 3: Evaluate & Generate
    evaluate_and_generate(
        model, val_loader, word2idx, idx2word, sequence_length, temperature, device
    )

if __name__ == "__main__":
    # Example usage:
    #
    # from data_processing import process_file
    # df_train = process_file("/content/lyrics_train_set.csv")
    #
    # main(df_train)
    #
    # Adjust the arguments as needed.
    pass
