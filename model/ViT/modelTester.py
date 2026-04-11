from torchvision import datasets, transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import timm
import re


#############################################################################################################################
#Important paths

#model must be in model/ViT/_____.pth
MODELFP = 'setToModelFilePath.pth' 
IMAGEPATH = 'setToPathofImageToBeClassified.jpg'

#############################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Image preprocess
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

#Removes number from class name. ex: "Apple 1" -> "Apple"
def strip_trailing_number(s):
    return re.sub(r'\s*\d+$', '', s).strip()

#Removes numbers from class names and condenses class groupings. 
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

#Takes an image and uses desired model and produces a classification. Outputs determined probability of predictions.
def predict_image(model, image_path, class_names, device):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    
    tensor = preprocess(img).unsqueeze(0).to(device)  # add batch dim
    
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

#Loads testing dataset to parse for classnames. Change dataset to match the model being used
testDS = datasets.ImageFolder("model/data/fruits-360/Test", transform=preprocess)
coarse_labels = sorted(set(strip_trailing_number(c) for c in testDS.classes))
coarse_to_idx = {c: i for i, c in enumerate(coarse_labels)}
remap_dataset(testDS, coarse_to_idx)

#Rebuilds desired model from previously trained models.
model = timm.create_model("vit_base_patch16_224",
                               pretrained=False,
                               img_size=100,
                               patch_size=10,
                               num_classes=len(testDS.classes))
model.load_state_dict(torch.load(f"model/ViT/{MODELFP}", map_location=device))
model.to(device)

#Creates prediction.
predict_image(model, IMAGEPATH, testDS.classes, device)