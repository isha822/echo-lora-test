import torch

def get_boundary_positions(labels):
    """
    Returns the index of the last prompt token for each sample in the batch.
    
    Args:
        labels: (batch, seq_len) tensor with -100 for prompt positions
    Returns:
        (batch,) tensor of boundary indices
    """
    mask = (labels == -100)
    return mask.sum(dim=1) -1

def get_answer_mask(labels):
    """
    Returns a float mask where 1.0 = answer token, 0.0 = prompt token.
    
    Args:
        labels: (batch, seq_len) tensor with -100 for prompt positions
    Returns:
        (batch, seq_len) float tensor
    """
    return (labels != -100).float()


def extract_boundary_hidden(
    hidden_state: torch.Tensor,
    boundary_position: torch.Tensor
) -> torch.Tensor:
    
    """
    Extracts the hidden state at the boundary for each sample.

    Args:
        hidden_state: (batch, seq_len, hidden_dim)
        boundary_positions: (batch,) indices from get_boundary_positions
    Returns:
        (batch, hidden_dim) boundary token representation
    """
    batch_size = hidden_state.shape[0]
    return hidden_state[torch.arange(batch_size), boundary_position, :]