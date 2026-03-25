import torch
import torch.nn as nn
from transformers import ViTModel

class ViT_LSTM_Captioner(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(ViT_LSTM_Captioner, self).__init__()
        
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.encoder_linear = nn.Linear(768, hidden_size)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        vit_outputs = self.encoder(pixel_values=images)
        features = vit_outputs.pooler_output 
        features = self.encoder_linear(features) 
        captions = captions[:, :-1]
        
        embeddings = self.embedding(captions)
        
        batch_size = features.size(0)
        h0 = features.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        
        outputs = self.linear(lstm_out) 
        return outputs

    def generate_caption(self, image, max_length=20, start_token_idx=1, end_token_idx=2):
        caption = []
        
        with torch.no_grad():
            vit_outputs = self.encoder(pixel_values=image)
            features = self.encoder_linear(vit_outputs.pooler_output)
            
            states = (features.unsqueeze(0), torch.zeros_like(features.unsqueeze(0)))
            
            inputs = torch.tensor([[start_token_idx]]).to(image.device)
            
            for _ in range(max_length):
                embeddings = self.embedding(inputs)
                
                hiddens, states = self.lstm(embeddings, states)
                
                outputs = self.linear(hiddens.squeeze(1))
                
                predicted_word_idx = outputs.argmax(1)
                
                if predicted_word_idx.item() == end_token_idx:
                    break
                    
                caption.append(predicted_word_idx.item())
                
                inputs = predicted_word_idx.unsqueeze(0)
                
        return caption
