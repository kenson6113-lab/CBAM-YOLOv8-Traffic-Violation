import os
import glob
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
    dataset_img_dir = "/content/master_traffic_violation_dataset/train/images"
    image_files = glob.glob(os.path.join(dataset_img_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(dataset_img_dir, "*.[pP][nN][gG]"))

    if image_files:
        sample_img_path = image_files[0]

        cbam_block = CBAM(channels=64)
        dummy_tensor = torch.randn(1, 64, 80, 80)
        cbam_output = cbam_block(dummy_tensor)
        print(f"CBAM Output Shape: {cbam_output.shape}")

        model = YOLO("yolov8n.pt")
        results = model.predict(source=sample_img_path, save=False)

        annotated_frame = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print(f"Error: No image files found in {dataset_img_dir}")
