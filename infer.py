import torch
from torchvision import models, transforms
from PIL import Image
import argparse

class_labels = [
    'Anthracnose',
    'Bacterial Wilt',
    'Belly Rot',
    'Downy Mildew',
    'Fresh Cucumber',
    'Fresh Leaf',
    'Gummy Stem Blight',
    'Pythium Fruit Rot'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, device):
    model = models.resnet152(weights=None)

    model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    return model


def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0, predicted_class].item()

    return predicted_class.item(), confidence


def main():
    parser = argparse.ArgumentParser(description="Cucumber Disease Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to classify")
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to the saved model (default: model.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    predicted_class, confidence = predict(args.image_path, model, device)

    print(f"Predicted class: {class_labels[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
