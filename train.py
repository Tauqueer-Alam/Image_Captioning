import torch
import torch.nn as nn
import torch.optim as optim
from model import ViT_LSTM_Captioner

def train_model():
    print("Setting up the model...")
    VOCAB_SIZE = 5000
    
    model = ViT_LSTM_Captioner(vocab_size=VOCAB_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        list(model.lstm.parameters()) + 
        list(model.linear.parameters()) + 
        list(model.encoder_linear.parameters()), 
        lr=0.001
    )
    
    print("Starting Training Loop...")
    
    EPOCHS = 3
    
    for epoch in range(EPOCHS):
        model.train()
        
        dummy_images = torch.randn(8, 3, 224, 224).to(device)
        dummy_captions = torch.randint(0, VOCAB_SIZE, (8, 20)).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(dummy_images, dummy_captions)
        
        outputs = outputs.reshape(-1, VOCAB_SIZE)
        targets = dummy_captions[:, 1:].reshape(-1)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
        
    print("Training finished! (Note: This was simulated data. Add real data to train a real, working model.)")

if __name__ == "__main__":
    train_model()
