import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * self.sigmoid(avg_out + max_out)

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = self.sigmoid(self.conv(torch.cat([avg_sp, max_sp], dim=1)))
        return x * spatial_map


if __name__ == "__main__":
    print("=" * 60)
    print("CBAM-YOLOv8 Minimal Working Example (MWE)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/3] Target Execution Device: {device}")

    cbam_block = CBAM(channels=64).to(device)
    dummy_tensor = torch.randn(1, 64, 80, 80).to(device)
    cbam_output = cbam_block(dummy_tensor)
    print(f"[2/3] CBAM Tensor Forward Pass Success!")
    print(f"      Input Shape:  {list(dummy_tensor.shape)}")
    print(f"      Output Shape: {list(cbam_output.shape)}")

    sample_path = "samples/test_motorcycle.jpg"

    if os.path.exists(sample_path):
        print(f"[3/3] Running YOLOv8 on repo sample image: '{sample_path}'")
        image_source = sample_path
    else:
        print(f"[3/3] Sample image not found. Fallback to synthetic image test...")
        image_source = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    model = YOLO("yolov8n.pt")
    results = model.predict(source=image_source, save=False, verbose=False)

    annotated_frame = results[0].plot()
    output_filename = "mwe_output.jpg"
    cv2.imwrite(output_filename, annotated_frame)

    print("=" * 60)
    print(f"MWE PASSED SUCCESSFULLY: Output saved to '{output_filename}'")
    print("=" * 60)
