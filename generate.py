import torch
import torch.nn.functional as F

def generate_text(
    model, 
    start_tokens, 
    word2idx, 
    idx2word, 
    max_length=20, 
    temperature=1.0, 
    device="cpu"
):
    """
    Generate text using the trained model with a sampling-based approach.
    """
    model.eval()
    result = [idx2word[idx] for idx in start_tokens]  # Start with known tokens

    batch_size = 1  # Single sequence generation
    state_h, state_c = model.init_hidden(batch_size)
    state_h, state_c = state_h.to(device), state_c.to(device)

    current_input = torch.tensor([start_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_length):
            logits, (state_h, state_c) = model(current_input, state_h, state_c)
            logits = logits / temperature  # Adjust for "creativity"
            probs = F.softmax(logits, dim=1).squeeze(0)
            next_token_id = torch.multinomial(probs, 1).item()
            result.append(idx2word[next_token_id])

            current_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

    return " ".join(result)
