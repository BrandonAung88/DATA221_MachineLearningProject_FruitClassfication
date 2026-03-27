import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import timm
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

#print("Worker Initiated")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(f"Using device: {device}")
#print(f"Number of GPUs: {torch.cuda.device_count()}")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 3e-4
DECAY = 0.05
DROP_RATE = 0.1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

trainPreprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
preprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
scaler = torch.amp.GradScaler("cuda")


#print(f"Classes : {len(trainDS.classes)}")
#print(f"Train   : {len(trainDS):,}   |  Test: {len(testDS):,}")
#print(f"Classes : {trainDS.classes}")

# Training loop

def train(model, loader, optimizer, criterion, device):
    model.train()
    totalLoss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True) 
        
        with torch.autocast("cuda"):                                        # ← AMP forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()                          # ← scaled backward
        scaler.unscale_(optimizer)                             # ← unscale before clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)                                 # ← scaler-aware step
        scaler.update()       

        totalLoss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return totalLoss / total, correct / total
# Evaluation loop
def evaluate(model, loader, criterion, device):
    model.eval()
    totalLoss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast("cuda"):                                  # ← AMP here too
                outputs = model(images)
                loss = criterion(outputs, labels)

            totalLoss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return totalLoss / total, correct / total
if __name__ == "__main__":

    trainDS = datasets.ImageFolder("model/data/fruits-360/Training", transform=trainPreprocess)
    testDS = datasets.ImageFolder("model/data/fruits-360/Test", transform=preprocess)

    trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True, prefetch_factor=2)
    testLoader  = DataLoader(testDS,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True,persistent_workers=True, prefetch_factor=2)

    # Instantiate model
    model = timm.create_model("vit_base_patch16_224",pretrained=False,img_size=100,patch_size=10, num_classes=len(trainDS.classes))
    #model.head = nn.Linear(model.embed_dim, len(trainDS.classes))
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Training
    trainAccs, testAccs, trainLosses, testLosses = [], [], [], []
    bestAcc = 0
    trainStart = time.time()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS} started.")
        trainLoss, trainAcc = train(model, trainLoader, optimizer, criterion, device)
        testLoss, testAcc = evaluate(model, testLoader, criterion, device)
        scheduler.step()

        trainAccs.append(trainAcc)
        testAccs.append(testAcc)
        trainLosses.append(trainLoss)
        testLosses.append(testLoss)
        
        epoch_time = time.time() - epoch_start

        if testAcc > bestAcc:
            bestAcc = testAcc
            torch.save(model.state_dict(), "vitBestCheckpoint.pth")
            print('Best accuracy checkpoint saved.')

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f} | Epoch Time: {epoch_time:.1f}s")
    print(f"Epoch {((time.time() - trainStart)/60):.2f}m")

    torch.save(model.state_dict(), "vitFruits.pth")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(trainLosses, label="Train Loss")
    ax1.plot(testLosses,   label="Val Loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(trainAccs, label="Train Acc")
    ax2.plot(testAccs,   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()