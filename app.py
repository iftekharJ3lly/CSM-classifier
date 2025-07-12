import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import torchvision.models as models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.eval()


class_names = ["aki", "denji", "makima", "power"]


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        top_class = torch.argmax(probs).item()
        return class_names[top_class]


interface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text")
interface.launch(share=True)
