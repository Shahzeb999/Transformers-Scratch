#  • Shapes are noted as (batch, seq_len, hidden) etc.
#  • Comments explain exactly what each line does and why.
#  • Occasional "Notes:" call out subtle behaviors or potential tweaks.

import torch                              # Import the PyTorch tensor & autograd library
import torch.nn as nn                      # Shorthand import for neural network modules
import math                                # Python math for sqrt, log, etc.


class LayerNormalization(nn.Module):       # Custom LayerNorm module (pre-norm style is used elsewhere)

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()                 # Initialize base nn.Module internals
        self.eps = eps                     # Numerical stability constant added to denominator
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale (γ) per feature dim
        self.bias  = nn.Parameter(torch.zeros(features)) # Learnable shift (β) per feature dim

    def forward(self, x):
        # x has shape: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)            # Per-token mean over feature dim: (B, S, 1)
        std  = x.std(dim=-1, keepdim=True)             # Per-token std  over feature dim: (B, S, 1)
        # Note: torch.std uses unbiased=True by default. For LayerNorm, unbiased=False is also common.
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # Normalize, then scale & shift


class FeedForwardBlock(nn.Module):         # Position-wise feed-forward network (applied at each token)

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)       # First projection W1 (expand dims)
        self.dropout  = nn.Dropout(dropout)            # Dropout for regularization
        self.linear_2 = nn.Linear(d_ff, d_model)       # Second projection W2 (project back)

    def forward(self, x):
        # (B, S, d_model) -> ReLU(W1 x) -> Dropout -> W2 -> (B, S, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):          # Token embedding layer with √d_model scaling

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model   = d_model                        # Hidden size / embedding dimension
        self.vocab_size = vocab_size                    # Vocabulary size for the embedding table
        self.embedding = nn.Embedding(vocab_size, d_model)  # Learnable lookup table (V, d_model)

    def forward(self, x):
        # Inputs are token IDs: (B, S). Output embeddings: (B, S, d_model)
        # Scale by √d_model as in the original Transformer paper (helps variance stability).
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):       # Classic sinusoidal positional encodings (non-learnable)

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model                          # Hidden size (must match embeddings)
        self.seq_len = seq_len                          # Maximum sequence length supported
        self.dropout = nn.Dropout(dropout)              # Dropout after adding PE

        pe = torch.zeros(seq_len, d_model)              # Preallocate PE matrix: (S, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (S, 1)
        # Frequencies for even/odd dims (vector of length d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # Even dims use sine
        pe[:, 1::2] = torch.cos(position * div_term)    # Odd  dims use cosine
        pe = pe.unsqueeze(0)                            # Add batch dim -> (1, S, d_model)
        self.register_buffer('pe', pe)                  # Save as buffer (not a parameter)

    def forward(self, x):
        # Slice to current sequence length and add to token embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (B, S, d_model)
        return self.dropout(x)                           # Apply dropout after PE addition


class ResidualConnection(nn.Module):       # Pre-norm residual connection wrapper

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)              # Dropout on sublayer output
        self.norm    = LayerNormalization(features)     # LayerNorm before sublayer (pre-norm)

    def forward(self, x, sublayer):
        # Apply: x + Dropout( Sublayer( LayerNorm(x) ) )
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):  # Multi-head self/cross attention

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model                          # Model hidden size
        self.h = h                                      # Number of attention heads
        assert d_model % h == 0, "d_model is not divisible by h"  # Ensure equal head sizes

        self.d_k = d_model // h                         # Per-head key/query/value dim
        # Learned linear projections for Q, K, V, and output (no bias as in many implementations)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)              # Dropout on attention weights

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]                           # Per-head dimension
        # Scaled dot-product attention scores: (B, h, S_q, d_k) @ (B, h, d_k, S_k) -> (B, h, S_q, S_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Mask out positions where mask == 0 (e.g., padding or future tokens for causal mask)
            attention_scores.masked_fill_(mask == 0, -1e9)  # Large negative ~ -inf for softmax
        attention_scores = attention_scores.softmax(dim=-1) # Normalize over key positions
        if dropout is not None:
            attention_scores = dropout(attention_scores)    # Regularize attention probabilities
        # Weighted sum of values: (B, h, S_q, S_k) @ (B, h, S_k, d_k) -> (B, h, S_q, d_k)
        return (attention_scores @ value), attention_scores # Return context and attention probs

    def forward(self, q, k, v, mask):
        # Project inputs to Q, K, V (each: (B, S, d_model))
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        # Reshape to multi-head: (B, S, d_model) -> (B, S, h, d_k) -> (B, h, S, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key  .view(key  .shape[0], key  .shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention per head
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate heads: (B, h, S, d_k) -> (B, S, h, d_k) -> (B, S, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear projection back to d_model
        return self.w_o(x)


class EncoderBlock(nn.Module):             # One Transformer encoder layer (self-attn + FFN)

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Multi-head self-attention module
        self.feed_forward_block   = feed_forward_block    # Position-wise feed-forward network
        # Two residual paths: one for attention, one for FFN
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Residual 1: self-attention sublayer with pre-norm
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Residual 2: feed-forward sublayer with pre-norm
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):                   # Stack of N encoder blocks + final LayerNorm

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers                            # ModuleList of encoder blocks
        self.norm   = LayerNormalization(features)      # Final LayerNorm after stack

    def forward(self, x, mask):
        for layer in self.layers:                       # Sequentially pass through all layers
            x = layer(x, mask)
        return self.norm(x)                             # Normalize the final encoder output


class DecoderBlock(nn.Module):             # One Transformer decoder layer (masked self-attn, cross-attn, FFN)

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block  = self_attention_block   # Masked self-attention for target tokens
        self.cross_attention_block = cross_attention_block  # Cross-attention over encoder outputs
        self.feed_forward_block    = feed_forward_block     # Position-wise FFN
        # Three residual paths: self-attn, cross-attn, FFN
        self.residual_connections  = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Residual 1: masked self-attention (uses tgt_mask for causality)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Residual 2: cross-attention (queries from decoder, keys/values from encoder)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Residual 3: FFN
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):                   # Stack of N decoder blocks + final LayerNorm

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers                            # ModuleList of decoder blocks
        self.norm   = LayerNormalization(features)      # Final LayerNorm after stack

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:                       # Pass through all decoder layers
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)                             # Normalize the final decoder output


class ProjectionLayer(nn.Module):           # Output head: map hidden states -> vocabulary logits

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)      # Linear projection to vocab size

    def forward(self, x) -> None:
        # (B, S, d_model) -> (B, S, vocab_size)
        # Note: type hint says -> None, but this returns a Tensor; consider -> torch.Tensor
        return self.proj(x)


class Transformer(nn.Module):               # Full seq2seq model wrapper (encoder + decoder + heads)

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder                          # Encoder stack
        self.decoder = decoder                          # Decoder stack
        self.src_embed = src_embed                      # Source token embedding
        self.tgt_embed = tgt_embed                      # Target token embedding
        self.src_pos = src_pos                          # Source positional encoding
        self.tgt_pos = tgt_pos                          # Target positional encoding
        self.projection_layer = projection_layer        # Final LM head / classifier

    def encode(self, src, src_mask):
        # Embed + add positions, then run encoder: returns encoded source reps (B, S_src, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Embed + add positions for target, then run decoder using encoder outputs
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # Map decoder hidden states to vocabulary logits
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int, tgt_vocab_size: int,
    src_seq_len: int, tgt_seq_len: int,
    d_model: int = 512, N: int = 6, h: int = 8,
    dropout: float = 0.1, d_ff: int = 2048
) -> Transformer:
    # Create source & target embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding modules for max sequence lengths
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Build N encoder blocks (each with its own attention & FFN)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Build N decoder blocks (masked self-attn + cross-attn + FFN)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Stack encoder & decoder layers into modules
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Output projection layer to vocabulary
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Assemble the full Transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Xavier/Glorot uniform init for all weight tensors with dim > 1 (common for linear layers)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer  # Return the constructed model instance

# --------- Additional Notes / Tips ---------
# • Masks: src_mask and tgt_mask must be broadcastable to attention score shapes.
#   Typical shapes are (B, 1, 1, S_k) for padding masks and (B, 1, S_q, S_k) for causal masks.
# • Precision: Using -1e9 for masking works in fp32; for fp16/bfloat16 consider torch.finfo(x.dtype).min.
# • Normalization: This uses pre-norm (LayerNorm before sublayer), which generally stabilizes deep stacks.
# • Activations: ReLU in FFN is faithful to the paper; GELU is a popular alternative.
# • LayerNorm stats: For LayerNorm, many libraries compute variance and add eps before sqrt; here we add eps to std.
# • Type hint: ProjectionLayer.forward "-> None" should be "-> torch.Tensor" for correctness.
# • Positional encodings are buffers (not trained) and reused across batches.
