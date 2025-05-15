import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math

class LatexOCRModel(nn.Module):
    """
    End-to-end model pro převod obrázků matematických výrazů na LaTeX kód.
    Používá encoder-decoder architekturu založenou na transformeru pro encoder i decoder.
    Navíc obsahuje VGG-blok pro lepší extrakci vizuálních příznaků.
    """

    def __init__(self, encoder_dim=128, vocab_size=1000, embedding_dim=64,
                 dropout=0.5, num_transformer_layers=4, nhead=8,
                 height=80, max_width=2048):
        super(LatexOCRModel, self).__init__()
        
        self.height = height
        self.max_width = max_width
        self.encoder_dim = encoder_dim
        
        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # VGG-inspired block for feature extraction
        self.vgg = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),     # Input channel is 1 for grayscale
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Update patch parameters
        self.patch_size = 8
        self.patch_embed = nn.Sequential(
            nn.Conv2d(64, encoder_dim, kernel_size=3, padding=1),  # Maintain spatial dimensions
            nn.BatchNorm2d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=self.patch_size, stride=self.patch_size)  # Patch embedding
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Transformer decoder components
        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=encoder_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)
        
        # Output projection
        self.fc = nn.Linear(encoder_dim, vocab_size)

        # Weight initialization
        self._init_weights()

    def get_1d_sinusoidal_pos_embed(self, pos, dim):
        """Generate 1D sinusoidal positional embeddings."""
        assert dim % 2 == 0, f"Embedding dimension {dim} should be even"
        pos = pos.unsqueeze(-1)
        omega = torch.exp(torch.arange(0, dim, 2, device=pos.device) * (-math.log(10000.0) / dim))
        out = pos * omega
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        return emb

    def get_2d_sinusoidal_pos_embed(self, h, w, embed_dim):
        """
        Generate 2D sinusoidal positional embeddings.
        Args:
            h, w: height and width of the grid
            embed_dim: embedding dimension (must be even)
        """
        grid_h = torch.arange(h, device=self.embedding.weight.device)
        grid_w = torch.arange(w, device=self.embedding.weight.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        # Flatten the grids
        grid_h = grid_h.flatten()
        grid_w = grid_w.flatten()
        
        # Split dimension for height and width components
        dim = embed_dim // 2
        h_embed = self.get_1d_sinusoidal_pos_embed(grid_h, dim)
        w_embed = self.get_1d_sinusoidal_pos_embed(grid_w, dim)
        pos_embed = torch.cat([h_embed, w_embed], dim=-1)
        
        return pos_embed

    def get_decoder_pos_embed(self, length, dim):
        """Generate positional embeddings for decoder sequence."""
        pos = torch.arange(length, device=self.embedding.weight.device)
        return self.get_1d_sinusoidal_pos_embed(pos, dim)

    def _init_weights(self):
        """Inicializace vah modelu."""
        for name, param in self.named_parameters():
            if "fc" in name or "embedding" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
            # Add positional embedding initialization
            elif "pos_embed_weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)  # Common initialization for transformers

    def preprocess_image(self, image):
        """
        Preprocesses the input image to have fixed height while preserving aspect ratio.
        Also ensures the width is divisible by patch_size.
        
        Args:
            image: Input image of shape [batch_size, channels, height, width]
            
        Returns:
            Preprocessed image of shape [batch_size, channels, self.height, padded_width]
        """
        batch_size, channels, height, width = image.shape
        
        # First, resize to fixed height while preserving aspect ratio
        if height != self.height:
            scale_factor = self.height / height
            new_width = int(width * scale_factor)
            # Use bicubic interpolation for better quality
            image = F.interpolate(image, size=(self.height, new_width), mode='bicubic', align_corners=False)
        
        # Get current dimensions after resize
        _, _, _, current_width = image.shape
        
        # Calculate padding to make width divisible by patch_size
        remainder = current_width % self.patch_size
        if remainder != 0:
            padding_width = self.patch_size - remainder
            padding = (0, padding_width, 0, 0)  # pad_left, pad_right, pad_top, pad_bottom
            image = F.pad(image, padding, "constant", 0)
        
        # Check if width exceeds max_width
        _, _, _, padded_width = image.shape
        if padded_width > self.max_width:
            image = F.interpolate(image, size=(self.height, self.max_width), mode='bicubic', align_corners=False)
        
        # Ensure minimum width for transformer processing
        min_width = self.patch_size * 2  # At least 2 patches
        if padded_width < min_width:
            padding_width = min_width - padded_width
            padding = (0, padding_width, 0, 0)
            image = F.pad(image, padding, "constant", 0)
        
        # Normalize pixel values if not already normalized
        if channels == 1 and image.max() > 1:
            image = image / 255.0
            if self.normalize is not None:
                image = self.normalize(image)

        return image

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, captions, caption_lengths):
        """
        Forward pass.

        Args:
            images: Obrázky matematických výrazů [batch_size, 3, H, W] (variabilní rozměry)
            captions: Tokenizované LaTeX výrazy [batch_size, max_seq_length]
            caption_lengths: Délky skutečných caption [batch_size]

        Returns:
            outputs: Predikce pro každý token [batch_size, max_seq_length, vocab_size]
        """
        batch_size = images.size(0)
        device = images.device
        seq_length = captions.size(1)  # Get actual sequence length
        
        # Preprocess images to normalized size
        images = self.preprocess_image(images)

        # VGG block for initial feature extraction
        images = self.vgg(images)  # [batch_size, 64, H/2, W/2]

        # Patch embedding
        features = self.patch_embed(images)  # [batch_size, encoder_dim, H/(2*patch_size), W/(2*patch_size)]
        h, w = features.shape[2:]
        features = features.flatten(2).transpose(1, 2)  # [batch_size, num_patches, encoder_dim]
        
        # Add 2D positional embeddings to image features
        pos_embed = self.get_2d_sinusoidal_pos_embed(h, w, self.encoder_dim)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        features = features + pos_embed

        # Transformer encoder
        memory = self.transformer_encoder(features)  # [batch_size, num_patches, encoder_dim]

        # Prepare decoder inputs with positional embeddings
        tgt = self.embedding(captions)  # [batch_size, max_seq_length, encoder_dim]
        decoder_pos = self.get_decoder_pos_embed(seq_length, self.encoder_dim)
        decoder_pos = decoder_pos.unsqueeze(0).expand(batch_size, -1, -1)
        tgt = tgt + decoder_pos
        tgt = self.embedding_dropout(tgt)

        # Create attention mask for the actual sequence length
        tgt_mask = self.generate_square_subsequent_mask(seq_length).to(device)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )  # [batch_size, max_seq_length, encoder_dim]

        # Apply final linear layer
        output = self.fc(output)  # [batch_size, max_seq_length, vocab_size]

        return output

    def predict(self, image, max_length=150, start_token=1, end_token=2, beam_size=3):
        """
        Generování LaTeX sekvence pro obrázek s využitím beam search.
        """
        device = image.device
        with torch.no_grad():
            # Preprocess image
            image = self.preprocess_image(image)
            
            # VGG block and Patch embedding
            image = self.vgg(image)
            features = self.patch_embed(image)
            h, w = features.shape[2:]
            features = features.flatten(2).transpose(1, 2)  # [1, num_patches, encoder_dim]
            
            # Add 2D positional embeddings
            pos_embed = self.get_2d_sinusoidal_pos_embed(h, w, self.encoder_dim)
            pos_embed = pos_embed.unsqueeze(0)  # [1, num_patches, encoder_dim]
            features = features + pos_embed
            
            # Transformer encoder
            memory = self.transformer_encoder(features)

            if beam_size > 1:
                return self._beam_search(memory, max_length, start_token, end_token, beam_size, device)
            else:
                return self._greedy_decode(memory, max_length, start_token, end_token, device)

    def _beam_search(self, memory, max_length, start_token, end_token, beam_size, device):
        """Helper method for beam search decoding."""
        sequences = [([], 0.0)]
        memory = memory.expand(beam_size, -1, -1)  # Expand memory for beam search
        
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                if seq and seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue
                    
                # Prepare decoder input
                tgt = torch.LongTensor([start_token] + seq).to(device)
                tgt = self.embedding(tgt)
                seq_length = tgt.size(0)
                
                # Add positional embeddings
                decoder_pos = self.get_decoder_pos_embed(seq_length, self.encoder_dim)
                tgt = tgt + decoder_pos
                tgt = tgt.unsqueeze(1)  # [seq_len, 1, encoder_dim]
                
                # Create attention mask with proper shape
                tgt_mask = torch.zeros((seq_length, seq_length), device=device).fill_(float('-inf'))
                tgt_mask = torch.triu(tgt_mask, diagonal=1)
                
                # Decode
                output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
                predictions = self.fc(output[-1])
                
                # Get top k candidates
                log_probs = F.log_softmax(predictions, dim=-1)
                topk_probs, topk_tokens = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_seq = seq + [topk_tokens[0, i].item()]
                    new_score = score + topk_probs[0, i].item()
                    all_candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_size]
            
            # Early stopping if all sequences end with end_token
            if all(s[-1] == end_token for s, _ in sequences):
                break
        
        return sequences[0][0]  # Return the sequence with highest score

    def _greedy_decode(self, memory, max_length, start_token, end_token, device):
        """Helper method for greedy decoding."""
        generated_sequence = []
        
        for _ in range(max_length):
            # Prepare decoder input
            tgt = torch.LongTensor([start_token] + generated_sequence).to(device)
            tgt = self.embedding(tgt)
            seq_length = tgt.size(0)
            
            # Add positional embeddings
            decoder_pos = self.get_decoder_pos_embed(seq_length, self.encoder_dim)
            tgt = tgt + decoder_pos
            tgt = tgt.unsqueeze(1)  # [seq_len, 1, encoder_dim]
            
            # Create attention mask with proper shape
            tgt_mask = torch.zeros((seq_length, seq_length), device=device).fill_(float('-inf'))
            tgt_mask = torch.triu(tgt_mask, diagonal=1)
            
            # Decode
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            predictions = self.fc(output[-1])
            
            # Get next token
            _, predicted_token = predictions.max(1)
            next_token = predicted_token.item()
            generated_sequence.append(next_token)
            
            if next_token == end_token:
                break
                
        return generated_sequence