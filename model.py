import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LyricsLSTMModel(nn.Module):
    """LSTM-based model for lyrics generation."""
    def __init__(self, num_embeddings, embedding_dim, hidden_size, dropout_p=0.5, padding_idx=0):
        super(LyricsLSTMModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_embeddings)

    def init_hidden(self, batch_size):
        """
        Initialize hidden and cell states with zeros.
        """
        state_h = torch.zeros(1, batch_size, self.hidden_size)
        state_c = torch.zeros(1, batch_size, self.hidden_size)
        return state_h, state_c

    def forward(self, input_sequence, state_h, state_c):
        """
        Forward pass through the model.
        Args:
            input_sequence: Tensor of shape (batch_size, seq_length)
            state_h, state_c: Initial hidden and cell states
        Returns:
            logits: Output predictions of shape (batch_size, vocab_size)
            (state_h, state_c): Updated hidden and cell states
        """
        # Apply embedding layer
        embedded = self.embedding(input_sequence)

        # Apply LSTM layer
        lstm_out, (state_h, state_c) = self.lstm(embedded, (state_h, state_c))

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Apply fully connected layer to the last time step
        logits = self.fc(lstm_out[:, -1, :])
        return logits, (state_h, state_c)

    def topk_sampling(self, logits, topk=5):
        """
        Apply top-k sampling to select the next word.
        Args:
            logits: Output logits of shape (1, vocab_size)
            topk: Number of top words to consider for sampling
        Returns:
            sampled_index: Index of the sampled word
        """
        probs = F.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=topk, dim=1)
        indices = indices.squeeze(0).tolist()
        sampled_index = random.choice(indices)
        return sampled_index
