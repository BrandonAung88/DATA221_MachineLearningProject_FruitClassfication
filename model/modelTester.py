from torchvision import datasets, transforms
from PIL import Image
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
import timm
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
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
def predict_image(model, image_path, class_names, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # match your training size
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    tensor = transform(img).unsqueeze(0).to(device)  # add batch dim
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top5 = probs.topk(4)
    
    print("Predictions:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {class_names[idx]:20s} {prob.item()*100:.1f}%")
    
    # Show image with prediction
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[top5.indices[0]]} ({top5.values[0]*100:.1f}%)")
    plt.axis("off")
    plt.show()

testDS = datasets.ImageFolder("model/data/fruits-360/Test", transform=preprocess)

coarse_labels = sorted(set(strip_trailing_number(c) for c in testDS.classes))
coarse_to_idx = {c: i for i, c in enumerate(coarse_labels)}

remap_dataset(testDS, coarse_to_idx)

model = timm.create_model("vit_base_patch16_224",
                               pretrained=False,
                               img_size=100,
                               patch_size=10,
                               num_classes=len(testDS.classes))

model.load_state_dict(torch.load("projSet3Model-20E.pth", map_location=device))
model.to(device)

predict_image(model, "oj.jpg", testDS.classes, device)