# finetune_qdn.py (last ad-hoc version; replaced later by the native QueryDiscussNet script)
# Early debugging attempt, copied from the first chat (LEGENDARY_DL.pdf). Kept for the record.
# Lines that the PDF cut off at the page edge are marked [clipped].
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_fixed import VinDrDataset
import torchvision.models as models
class QDNModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.proj4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.proj5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.tok_norm = nn.LayerNorm(256)
        
        # Use ParameterList or ParameterDict for tensor parameters inside ModuleDict/Module
        self.step = nn.ModuleDict({
            'read': nn.MultiheadAttention(256, 8, batch_first=True),
            'talk': nn.MultiheadAttention(256, 8, batch_first=True),
            'ffn': nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 256)),
            'write': nn.MultiheadAttention(256, 8, batch_first=True)
        })
        self.step_params = nn.ParameterDict({
            'n_q': nn.Parameter(torch.randn(256)),
            'n_kv': nn.Parameter(torch.randn(256)),
            'n_s': nn.Parameter(torch.randn(256)),
            'n_f': nn.Parameter(torch.randn(256)),
            'n_t': nn.Parameter(torch.randn(256)),
            'n_qq': nn.Parameter(torch.randn(256))
        })
        
        self.out_norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        out = self.proj5(features).mean(dim=[2, 3])
        out = self.tok_norm(out)
        out = self.classifier(out)
        return out
def main():
    print("=== Executing Hard-Negative Fine-Tuning Training Loop ===")
    
    train_csv = "./splits/train.csv"
    img_dir = "./images"
    batch_size = 16
    epochs = 2
    lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VinDrDataset(csv_file=train_csv, img_dir=img_dir, transform=transform, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)  # [clipped] ending restored
    
    model = QDNModel(num_classes=len(train_dataset.label_cols)).to(device)
    
    checkpoint_path = "outputs/final/qdn/best_ema.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights successfully from {checkpoint_path}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    os.makedirs("outputs/final/qdn_hard_neg", exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")  # [clipped] ending restored
                
        print(f"Epoch {epoch+1} Complete. Average Loss: {total_loss / len(train_loader):.4f}")
        
    torch.save({"state_dict": model.state_dict()}, "outputs/final/qdn_hard_neg/best_ema.pt")
    print("Fine-tuning complete. Saved updated model to outputs/final/qdn_hard_neg/best_ema.pt")
if __name__ == "__main__":
    main()
