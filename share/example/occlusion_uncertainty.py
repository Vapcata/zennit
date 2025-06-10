import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def mc_prediction(model, image, forward_passes=10):
    enable_dropout(model)
    with torch.no_grad():
        preds = [model(image).cpu().numpy() for _ in range(forward_passes)]
    return np.vstack(preds)


def occlusion_sensitivity(model, image, target_class, window=4, stride=4, passes=10):
    c, h, w = image.shape
    occlusion_probs = []
    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            occluded = image.clone()
            occluded[:, y:y + window, x:x + window] = 0.0
            preds = mc_prediction(model, occluded.unsqueeze(0), forward_passes=passes)
            occlusion_probs.append(preds[:, target_class].mean())
    return np.array(occlusion_probs)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    model = SimpleNN()
    model.load_state_dict(torch.load("data/mnist_model.pth"))
    model.eval()

    uncertain_images = [1808, 881, 7764, 4536, 1235, 1337, 3172, 6034, 6428, 8253]

    for idx in uncertain_images:
        image = test_dataset[idx][0]
        base_preds = mc_prediction(model, image.unsqueeze(0))
        mean_probs = base_preds.mean(axis=0)
        target = int(mean_probs.argmax())

        occl_probs = occlusion_sensitivity(model, image, target)

        plt.figure(figsize=(8, 5))
        plt.boxplot([base_preds[:, target], occl_probs], labels=["original", "occluded"])
        plt.title(f"Image {idx} class {target}")
        plt.ylabel("Predicted probability")
        plt.grid(True)
        plt.savefig(f"results/occlusion_box_{idx}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
