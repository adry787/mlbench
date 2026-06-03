#!/usr/bin/env python3
"""MLBench - Model Benchmark."""
import torch
import torchvision.models as models
import time


class Benchmark:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.dev}")

    def time_inference(self, model, runs=50):
        model.to(self.dev).eval()
        x = torch.randn(1, 3, 224, 224).to(self.dev)
        for _ in range(10):
            model(x)
        start = time.time()
        for _ in range(runs):
            model(x)
        return round((time.time() - start) / runs * 1000, 2)

    def count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def compare(self, models_dict):
        print(f"\n{'Name':15s} {'Params':>12s} {'MS':>8s}")
        print("-" * 40)
        for name, model in models_dict.items():
            params = self.count_params(model)
            ms = self.time_inference(model)
            print(f"{name:15s} {params:>12,} {ms:>8}")


if __name__ == "__main__":
    b = Benchmark()
    models_dict = {
        "ResNet-18": models.resnet18(weights=None),
        "ResNet-50": models.resnet50(weights=None),
        "MobileNet": models.mobilenet_v2(weights=None),
    }
    b.compare(models_dict)
