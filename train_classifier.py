import torchvision
from torchvision import models


def export_onnx(model, save_path="weights/classifier/efficientnet_b1.onnx"):
    import torch
    dummy_input = torch.randn(1, 3, 240, 240)
    model.eval()
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"✅ Exported to {save_path}")

if __name__ == "__main__":
    model = models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
    export_onnx(model)
    print("ONNX export complete.")