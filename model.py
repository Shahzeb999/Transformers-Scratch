import torch # Import the PyTorch tensor & autograd library
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
decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
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


return transformer # Return the constructed model instance
