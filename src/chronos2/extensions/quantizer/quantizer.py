import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    The Q-Chronos Bottleneck. 
    Maps continuous signals to a 'vocabulary' of temporal primitives.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # The Codebook: A dictionary of learned 'shape' vectors.
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs: (batch, seq_len, d_model)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate L2 distances between input patches and codebook entries
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
            
        # Find the index of the closest temporal primitive
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Map indices back to codebook vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.codebook.weight).view(inputs.shape)
        
        # VQ Losses (Codebook + Commitment)
        codebook_loss = F.mse_loss(quantized.detach(), inputs) 
        commitment_loss = F.mse_loss(quantized, inputs.detach()) 
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-Through Estimator: copy gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices