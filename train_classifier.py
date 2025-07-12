import os
import torchvision
from torchvision import models, transforms
import torch
from going_modular import data_setup, engine
from pathlib import Path

NUM_WORKERS = os.cpu_count()
# Setup path to data folder
data_path = Path("dataset")
image_path = data_path / "biclassifier_v1"
train_dir = image_path / "train"
test_dir = image_path / "test"

data_transform = transforms.Compose([   
    transforms.Resize((260, 260)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=10),     # Small rotations simulate camera variability
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Simulates lighting changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                            train_dir=train_dir,
                            test_dir=test_dir,
                            transform=data_transform,
                            batch_size=3,
                            num_workers=4)
# Check out single image size/shape
img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}",class_names)

# Initialize model
model = models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
# Modify the classifier to output 2 classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Setup optimizer and loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model using the engine
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=200,
             device=device)

# Save the trained model weights
model_save_path = "weights/biclassifier/efficientnet_b1_trained.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Trained model saved to {model_save_path}")

def export_onnx(model, save_path="weights/biclassifier/efficientnet_b1_trained.onnx"):
    dummy_input = torch.randn(1, 3, 260, 260)
    model.eval()
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"✅ Exported to {save_path}")


class ExportONNX:
    def __init__(self):
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        weight_path = "weights/biclassifier/efficientnet_b1_trained.pth"
        model.load_state_dict(torch.load(weight_path))
        export_onnx(model)
        print("ONNX export complete.")
        
ExportONNX()