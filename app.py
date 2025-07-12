import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )
        self.block5 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=8410, out_features=4)
        )
    def forward(self, x):
        return self.block5(self.block4(self.block3(self.block2(self.block1(x)))))

model = ConvolutionalNN()
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
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
interface.launch()
