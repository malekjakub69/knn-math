import torch
import torch.nn as nn


class LatexOCRModel(nn.Module):
    """
    End-to-end model pro převod obrázků matematických výrazů na LaTeX kód.
    Používá encoder-decoder architekturu s attention mechanismem.
    """

    def __init__(self, encoder_dim=256, decoder_dim=512, vocab_size=1000, embedding_dim=256, attention_dim=256, dropout=0.5):
        super(LatexOCRModel, self).__init__()

        # Encoder - CNN pro extrakci vizuálních příznaků
        self.encoder = self._build_encoder(encoder_dim)

        # Attention mechanismus
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Decoder - LSTM pro generování LaTeX sekvence
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim + encoder_dim, decoder_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Inicializace vah
        self._init_weights()

    def _build_encoder(self, encoder_dim):
        """
        Vytvoření CNN enkodéru založeného na ResNet-18 s úpravami pro LaTeX OCR.
        """
        # Použijeme předtrénovaný ResNet-18 jako backbone
        resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)

        # Odstraníme poslední plně propojenou vrstvu a pooling
        modules = list(resnet.children())[:-2]

        # Přidáme 1x1 konvoluci pro snížení dimenze
        modules.append(nn.Conv2d(512, encoder_dim, kernel_size=1))
        modules.append(nn.BatchNorm2d(encoder_dim))
        modules.append(nn.ReLU(inplace=True))

        return nn.Sequential(*modules)

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
            images: Obrázky matematických výrazů [batch_size, 3, H, W]
            captions: Tokenizované LaTeX výrazy [batch_size, max_seq_length]
            caption_lengths: Délky skutečných caption [batch_size]

        Returns:
            outputs: Predikce pro každý token [batch_size, max_seq_length, vocab_size]
        """
        batch_size = images.size(0)

        # Encoder - získání vizuálních příznaků
        features = self.encoder(images)  # [batch_size, encoder_dim, H', W']
        features_dim = features.size(1)
        features_h, features_w = features.size(2), features.size(3)

        # Flatten spatial dimensions
        features = features.permute(0, 2, 3, 1).contiguous()  # [batch_size, H', W', encoder_dim]
        features = features.view(batch_size, -1, features_dim)  # [batch_size, H'*W', encoder_dim]

        # Decoder s attention
        embeddings = self.embedding(captions)  # [batch_size, max_seq_length, embedding_dim]

        # Inicializace LSTM hidden a cell stavu
        h = torch.zeros(1, batch_size, self.decoder.hidden_size).to(images.device)
        c = torch.zeros(1, batch_size, self.decoder.hidden_size).to(images.device)

        # Dekódování sekvence
        max_length = captions.size(1)
        outputs = torch.zeros(batch_size, max_length, self.fc.out_features).to(images.device)

        for t in range(max_length):
            # Attention mechanismus
            context, _ = self.attention(features, h.squeeze(0))

            # LSTM input v čase t
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)

            # LSTM forward pass
            lstm_out, (h, c) = self.decoder(lstm_input, (h, c))

            # Predikce distribuce pravděpodobnosti
            predictions = self.fc(self.dropout(lstm_out.squeeze(1)))
            outputs[:, t, :] = predictions

        return outputs

    def predict(self, image, max_length=150, start_token=1, end_token=2):
        """
        Generování LaTeX sekvence pro obrázek.
        """
        with torch.no_grad():
            # Enkódování obrázku
            features = self.encoder(image)  # [1, encoder_dim, H', W']
            features_dim = features.size(1)
            features = features.permute(0, 2, 3, 1).contiguous()  # [1, H', W', encoder_dim]
            features = features.view(1, -1, features_dim)  # [1, H'*W', encoder_dim]

            # Inicializace LSTM hidden a cell stavu
            h = torch.zeros(1, 1, self.decoder.hidden_size).to(image.device)
            c = torch.zeros(1, 1, self.decoder.hidden_size).to(image.device)

            # První token je start token
            curr_token = torch.LongTensor([start_token]).to(image.device)

            # Postupné generování sekvence
            generated_sequence = []

            for _ in range(max_length):
                # Embedding aktuálního tokenu
                embedding = self.embedding(curr_token).unsqueeze(1)  # [1, 1, embedding_dim]

                # Attention
                context, _ = self.attention(features, h.squeeze(0))

                # LSTM input
                lstm_input = torch.cat([embedding.squeeze(1), context], dim=1).unsqueeze(1)

                # LSTM forward pass
                lstm_out, (h, c) = self.decoder(lstm_input, (h, c))

                # Predikce distribuce pravděpodobnosti
                predictions = self.fc(lstm_out.squeeze(1))

                # Výběr tokenu s nejvyšší pravděpodobností
                _, predicted_token = predictions.max(1)

                # Přidání do vygenerované sekvence
                generated_sequence.append(predicted_token.item())

                # Kontrola ukončení sekvence
                if predicted_token.item() == end_token:
                    break

                # Aktualizace tokenu pro další krok
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

    def forward(self, encoder_features, decoder_hidden):
        """
        Forward pass.

        Args:
            encoder_features: Vizuální příznaky z encoderu [batch_size, num_pixels, encoder_dim]
            decoder_hidden: Hidden state z decoderu [batch_size, decoder_dim]

        Returns:
            context: Kontextový vektor [batch_size, encoder_dim]
            alpha: Attention weights [batch_size, num_pixels]
        """
        # Výpočet attention score
        att1 = self.encoder_att(encoder_features)  # [batch_size, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size, attention_dim]

        # Součet s rozšířeným decoder state
        att = torch.tanh(att1 + att2.unsqueeze(1))  # [batch_size, num_pixels, attention_dim]

        # Výpočet attention weights
        att = self.full_att(att).squeeze(2)  # [batch_size, num_pixels]
        alpha = torch.softmax(att, dim=1)  # [batch_size, num_pixels]

        # Výpočet kontextového vektoru
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size, encoder_dim]

        return context, alpha
