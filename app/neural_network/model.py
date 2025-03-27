import torch
import torch.nn as nn

class LatexOCRModel(nn.Module):
    """
    End-to-end model pro převod obrázků matematických výrazů na LaTeX kód.
    Používá encoder-decoder architekturu; nyní se encoder zakládá na transformeru.
    Navíc obsahuje VGG-blok pro lepší extrakci vizuálních příznaků.
    """

    def __init__(self, encoder_dim=256, decoder_dim=512, vocab_size=1000, embedding_dim=256,
                 attention_dim=256, dropout=0.5, num_transformer_layers=6, nhead=8):
        super(LatexOCRModel, self).__init__()

        # VGG-inspired block for feature extraction
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces image from 224x224 to 112x112
        )

        # Update patch parameters after VGG: new image resolution is 112x112
        self.patch_size = 16
        self.num_patches = (112 // self.patch_size) ** 2  # 7x7 = 49
        self.encoder_dim = encoder_dim

        # Patch embedding using a convolutional layer.
        # Note: input channels now match the VGG output (64).
        self.patch_embed = nn.Conv2d(64, encoder_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # Learnable positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Attention mechanism
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Decoder - LSTM for generating LaTeX sequences
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(embedding_dim + encoder_dim, decoder_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Batch normalization for better training
        self.bn = nn.BatchNorm1d(decoder_dim)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Inicializace vah modelu."""
        for name, param in self.named_parameters():
            if "fc" in name or "decoder" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, images, captions, caption_lengths):
        """
        Forward pass.

        Args:
            images: Obrázky matematických výrazů [batch_size, 3, H, W] (očekáváme H=W=224)
            captions: Tokenizované LaTeX výrazy [batch_size, max_seq_length]
            caption_lengths: Délky skutečných caption [batch_size]

        Returns:
            outputs: Predikce pro každý token [batch_size, max_seq_length, vocab_size]
        """
        batch_size = images.size(0)

        # VGG block for initial feature extraction
        images = self.vgg(images)  # [batch_size, 64, 112, 112]

        # Patch embedding
        features = self.patch_embed(images)  # [batch_size, encoder_dim, 7, 7]
        features = features.flatten(2).transpose(1, 2)  # [batch_size, num_patches, encoder_dim]
        features = features + self.pos_emb  # Add positional embeddings

        # Transformer encoder expects shape [num_patches, batch_size, encoder_dim]
        features = features.transpose(0, 1)  # [num_patches, batch_size, encoder_dim]
        features = self.transformer_encoder(features)  # [num_patches, batch_size, encoder_dim]
        features = features.transpose(0, 1)  # [batch_size, num_patches, encoder_dim]

        # Decoder with attention
        embeddings = self.embedding(captions)  # [batch_size, max_seq_length, embedding_dim]
        embeddings = self.embedding_dropout(embeddings)

        # Initialize LSTM hidden and cell state
        h = torch.zeros(1, batch_size, self.decoder.hidden_size).to(images.device)
        c = torch.zeros(1, batch_size, self.decoder.hidden_size).to(images.device)

        max_length = captions.size(1)
        outputs = torch.zeros(batch_size, max_length, self.fc.out_features).to(images.device)

        for t in range(max_length):
            # Apply attention
            context, _ = self.attention(features, h.squeeze(0))

            # LSTM input for time t
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)
            lstm_out, (h, c) = self.decoder(lstm_input, (h, c))

            lstm_out_flat = lstm_out.squeeze(1)
            if lstm_out_flat.size(0) > 1:
                lstm_out_norm = self.bn(lstm_out_flat)
            else:
                lstm_out_norm = lstm_out_flat

            predictions = self.fc(self.dropout(lstm_out_norm))
            outputs[:, t, :] = predictions

        return outputs

    def predict(self, image, max_length=150, start_token=1, end_token=2, beam_size=3):
        """
        Generování LaTeX sekvence pro obrázek s využitím beam search.
        """
        with torch.no_grad():
            # VGG block and Patch embedding & Transformer encoder
            image = self.vgg(image)  # [1, 64, 112, 112]
            features = self.patch_embed(image)  # [1, encoder_dim, 7, 7]
            features = features.flatten(2).transpose(1, 2)  # [1, num_patches, encoder_dim]
            features = features + self.pos_emb
            features = features.transpose(0, 1)
            features = self.transformer_encoder(features)
            features = features.transpose(0, 1)

            h = torch.zeros(1, 1, self.decoder.hidden_size).to(image.device)
            c = torch.zeros(1, 1, self.decoder.hidden_size).to(image.device)
            curr_token = torch.LongTensor([start_token]).to(image.device)

            if beam_size > 1:
                sequences = [([], 0.0, h, c)]
                for _ in range(max_length):
                    all_candidates = []
                    for seq, score, h_prev, c_prev in sequences:
                        if seq and seq[-1] == end_token:
                            all_candidates.append((seq, score, h_prev, c_prev))
                            continue
                        last_token = torch.LongTensor([seq[-1] if seq else start_token]).to(image.device)
                        embedding = self.embedding(last_token)
                        context, _ = self.attention(features, h_prev.squeeze(0))
                        lstm_input = torch.cat([embedding, context], dim=1).unsqueeze(1)
                        lstm_out, (h_next, c_next) = self.decoder(lstm_input, (h_prev, c_prev))
                        if lstm_out.size(0) > 1:
                            lstm_out_norm = self.bn(lstm_out.squeeze(1))
                        else:
                            lstm_out_norm = lstm_out.squeeze(1)
                        predictions = self.fc(self.dropout(lstm_out_norm))
                        probs = torch.nn.functional.log_softmax(predictions, dim=1)
                        topk_probs, topk_tokens = probs.topk(beam_size)
                        for i in range(beam_size):
                            new_seq = seq + [topk_tokens[0, i].item()]
                            new_score = score + topk_probs[0, i].item()
                            all_candidates.append((new_seq, new_score, h_next, c_next))
                    all_candidates.sort(key=lambda x: x[1], reverse=True)
                    sequences = all_candidates[:beam_size]
                    if all(seq[-1] == end_token for seq, _, _, _ in sequences):
                        break
                best_seq = sequences[0][0]
                return best_seq

            else:
                generated_sequence = []
                for _ in range(max_length):
                    embedding = self.embedding(curr_token)
                    context, _ = self.attention(features, h.squeeze(0))
                    lstm_input = torch.cat([embedding, context], dim=1).unsqueeze(1)
                    lstm_out, (h, c) = self.decoder(lstm_input, (h, c))
                    predictions = self.fc(lstm_out.squeeze(1))
                    _, predicted_token = predictions.max(1)
                    generated_sequence.append(predicted_token.item())
                    if predicted_token.item() == end_token:
                        break
                    curr_token = predicted_token
                return generated_sequence


class Attention(nn.Module):
    """
    Attention mechanismus pro zaměření na relevantní části obrázku.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_features, decoder_hidden):
        """
        Forward pass.

        Args:
            encoder_features: Vizuální příznaky z encoderu [batch_size, num_tokens, encoder_dim]
            decoder_hidden: Hidden state z decoderu [batch_size, decoder_dim]

        Returns:
            context: Kontextový vektor [batch_size, encoder_dim]
            alpha: Attention weights [batch_size, num_tokens]
        """
        att1 = self.encoder_att(encoder_features)
        att2 = self.decoder_att(decoder_hidden)
        att = self.relu(att1 + att2.unsqueeze(1))
        att = self.full_att(att).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha