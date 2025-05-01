import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class LatexOCRModel(nn.Module):
    """
    End-to-end model pro převod obrázků matematických výrazů na LaTeX kód.
    Používá encoder-decoder architekturu založenou na transformeru pro encoder i decoder.
    Navíc obsahuje VGG-blok pro lepší extrakci vizuálních příznaků.
    """

    def __init__(self, encoder_dim=128, vocab_size=1000, embedding_dim=64,
                 dropout=0.5, num_transformer_layers=4, nhead=8,
                 height=80, max_width=2048):  # Removed unnecessary parameters
        super(LatexOCRModel, self).__init__()
        
        self.height = height
        self.max_width = max_width
        
        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # VGG-inspired block for feature extraction
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces height from 80 to 40
        )

        # Update patch parameters: height is fixed, width is variable
        self.patch_size = 8
        self.patch_height = height // 2 // self.patch_size  # 40 // 8 = 5
        self.encoder_dim = encoder_dim

        # Patch embedding using a convolutional layer
        self.patch_embed = nn.Conv2d(64, encoder_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Adaptive positional embeddings for encoder
        max_patches = (self.height // 2 // self.patch_size) * (self.max_width // 2 // self.patch_size)
        self.pos_embed_weight = nn.Parameter(torch.zeros(1, max_patches, encoder_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Transformer decoder components
        self.embedding = nn.Embedding(vocab_size, encoder_dim)  # Changed embedding_dim to encoder_dim for compatibility
        self.embedding_dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=encoder_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)
        
        # Output projection
        self.fc = nn.Linear(encoder_dim, vocab_size)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Inicializace vah modelu."""
        for name, param in self.named_parameters():
            if "fc" in name or "embedding" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def preprocess_image(self, image):
        """
        Preprocesses the input image to have fixed height while preserving aspect ratio.
        Also pads width to be divisible by patch_size.
        
        Args:
            image: Input image of shape [batch_size, channels, height, width]
            
        Returns:
            Preprocessed image of shape [batch_size, channels, self.height, padded_width]
        """
        batch_size, channels, height, width = image.shape
        
        # Resize to fixed height while preserving aspect ratio
        if height != self.height:
            scale_factor = self.height / height
            new_width = int(width * scale_factor)
            image = F.interpolate(image, size=(self.height, new_width), mode='bilinear', align_corners=False)
        
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
            # If image is too wide, resize it to max width
            image = F.interpolate(image, size=(self.height, self.max_width), mode='bilinear', align_corners=False)
        
        # Normalize pixel values
        if channels == 3:  # Only normalize RGB images
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
        features = features.flatten(2).transpose(1, 2)  # [batch_size, num_patches, encoder_dim]
        
        # Create positional embedding for the actual number of patches
        num_patches = features.shape[1]
        pos_emb = self.pos_embed_weight[:, :num_patches, :]
        features = features + pos_emb  # Add positional embeddings

        # Transformer encoder
        memory = self.transformer_encoder(features)  # [batch_size, num_patches, encoder_dim]

        # Prepare decoder inputs
        tgt = self.embedding(captions)  # [batch_size, max_seq_length, encoder_dim]
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
            
            # VGG block and Patch embedding & Transformer encoder
            image = self.vgg(image)  # [1, 64, H/2, W/2]
            features = self.patch_embed(image)  # [1, encoder_dim, H/(2*patch_size), W/(2*patch_size)]
            features = features.flatten(2).transpose(1, 2)  # [1, num_patches, encoder_dim]
            
            # Create positional embedding for the actual number of patches
            num_patches = features.shape[1]
            pos_emb = self.pos_embed_weight[:, :num_patches, :]
            features = features + pos_emb
            
            # Transformer encoder
            features = features.transpose(0, 1)  # [num_patches, 1, encoder_dim]
            memory = self.transformer_encoder(features)

            if beam_size > 1:
                return self._beam_search(memory, max_length, start_token, end_token, beam_size, device)
            else:
                return self._greedy_decode(memory, max_length, start_token, end_token, device)

    def _beam_search(self, memory, max_length, start_token, end_token, beam_size, device):
        """Helper method for beam search decoding."""
        sequences = [([], 0.0)]
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                if seq and seq[-1] == end_token:
                    all_candidates.append((seq, score))
                    continue
                    
                # Prepare decoder input
                tgt = torch.LongTensor([start_token] + seq).to(device)
                tgt = self.embedding(tgt).unsqueeze(1)  # [seq_len, 1, encoder_dim]
                tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)
                
                # Decode
                output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
                predictions = self.fc(output[-1])  # Get predictions for next token
                
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
        curr_token = torch.LongTensor([start_token]).to(device)
        
        for _ in range(max_length):
            # Prepare decoder input
            tgt = torch.LongTensor([start_token] + generated_sequence).to(device)
            tgt = self.embedding(tgt).unsqueeze(1)  # [seq_len, 1, encoder_dim]
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)
            
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