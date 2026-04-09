import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import re
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters

#Boolean to determine if user wants to continue training an existing model (True) or train new model from scratch (False).
FROM_CHECKPOINT = True
#Filepath to existing model - Same file path is used to choose model used in validation mode.
CKPTFILE = 'projSet4Model-20E.pth'

SAVEFILE = 'projSet4'
VALIDATE = True


BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 3e-4
DECAY = 0.05
DROP_RATE = 0.1
USE_BF16 = False
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


trainPreprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
scaler = torch.amp.GradScaler("cuda", enabled=not USE_BF16)
###################################################################################################################################   
def strip_trailing_number(s):
    return re.sub(r'\s*\d+$', '', s).strip()

def remap_dataset(ds, coarse_to_idx):
    original_classes = ds.classes  # save before overwriting
    fine_to_coarse = {
        idx: coarse_to_idx[strip_trailing_number(cls)]
        for idx, cls in enumerate(original_classes)
    }
    ds.targets = [fine_to_coarse[t] for t in ds.targets]
    ds.samples = [(p, fine_to_coarse[t]) for p, t in ds.samples]
    ds.classes = sorted(coarse_to_idx, key=coarse_to_idx.get)  # ← fixes .classes
    ds.class_to_idx = coarse_to_idx

######################################################################################################################################  
# Training loop
def train(model, loader, optimizer, criterion, device):
    
    model.train()
    totalLoss, correct, total = 0, 0, 0
    data_time, compute_time = 0, 0
    t0 = time.time()
    for images, labels in loader:
        data_time += time.time() - t0
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        t1 = time.time()
        optimizer.zero_grad(set_to_none=True) 
        
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if USE_BF16:
            loss.backward()                                   # bf16: no scaler needed
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        compute_time += time.time() - t1
        t0 = time.time()

        totalLoss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    print(f"Data: {data_time:.1f}s | Compute: {compute_time:.1f}s")
    return totalLoss / total, correct / total
# Evaluation loop
def evaluate(model, loader, criterion, device):
    model.eval()
    totalLoss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast("cuda", dtype=AMP_DTYPE):                              # ← AMP here too
                outputs = model(images)
                loss = criterion(outputs, labels)

            totalLoss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return totalLoss / total, correct / total

def validate(model, loader, criterion, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    totalLoss, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                outputs = model(images)
                loss = criterion(outputs, labels)

            totalLoss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += images.size(0)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    print(f"\nVal Loss: {totalLoss/total:.4f} | Val Acc: {acc:.4f}")

    # Per-class breakdown
    print("\n--- Per Class Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    return all_preds, all_labels

if __name__ == "__main__":
    
    print(f"Using device: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if not VALIDATE:

        trainDS = datasets.ImageFolder("../../data/fruits-360/Training", transform=trainPreprocess)
        testDS = datasets.ImageFolder("../../data/fruits-360/Test", transform=preprocess)

        coarse_labels = sorted(set(strip_trailing_number(c) for c in trainDS.classes))
        coarse_to_idx = {c: i for i, c in enumerate(coarse_labels)}

        remap_dataset(trainDS, coarse_to_idx)
        remap_dataset(testDS, coarse_to_idx)

        trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True, prefetch_factor=4)
        testLoader  = DataLoader(testDS,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True,persistent_workers=True, prefetch_factor=4)

        print(f"Classes : {len(trainDS.classes)}")
        print(f"Train   : {len(trainDS):,}   |  Test: {len(testDS):,}")
        print(f"Classes : {trainDS.classes}")
   
    # Instantiate model
        model = timm.create_model("vit_base_patch16_224",
                                pretrained=False,
                                img_size=100,
                                patch_size=10,
                                drop_rate=DROP_RATE,
                                num_classes=len(trainDS.classes))
        if FROM_CHECKPOINT:
            checkpoint = torch.load(f"{CKPTFILE}", map_location=device)
            filtered = {k: v for k, v in checkpoint.items() if not k.startswith("head.")}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
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
                
                torch.save(model.state_dict(), f"{SAVEFILE}bestCKPT-{EPOCHS}E.pth")

                print('Best accuracy checkpoint saved.')
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {trainLoss:.4f}, Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f} | Epoch Time: {epoch_time:.1f}s")

        print(f"Total test: {((time.time() - trainStart)/60):.2f}m")

        torch.save(model.state_dict(), f"{SAVEFILE}Model-{EPOCHS}E.pth")

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

    else:

        validateDS = datasets.ImageFolder("../data/fruits-360/Test", transform=trainPreprocess)
        coarse_labels = sorted(set(strip_trailing_number(c) for c in validateDS.classes))
        coarse_to_idx = {c: i for i, c in enumerate(coarse_labels)}

        remap_dataset(validateDS, coarse_to_idx)

        validateLoader = DataLoader(validateDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True,persistent_workers=True, prefetch_factor=4)


        model = timm.create_model("vit_base_patch16_224",
                               pretrained=False,
                               img_size=100,
                               patch_size=10,
                               num_classes=len(validateDS.classes))
        model.load_state_dict(torch.load(f"{CKPTFILE}", map_location=device))
        model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        preds, labels = validate(model, validateLoader, criterion, device, validateDS.classes)
