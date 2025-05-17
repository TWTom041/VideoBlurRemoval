import torch
from torcheval.metrics import PeakSignalNoiseRatio
import torchvision.transforms as transforms
import os
import cv2

metric = PeakSignalNoiseRatio()
vid="98_zPGqEMMWyx4"
ori_path=f"/home/twtomtwcc00/VSPW/data/{vid}/origin"
res_path="./deblur_test_out.mp4"


cv2.VideoCapture()

cap = cv2.VideoCapture(res_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transforms.ToTensor()(frame).unsqueeze(0)  # (1, 3, H, W)
    frames.append(frame)
cap.release()
res_vid=torch.cat(frames, dim=0)

frames=[]
for i in sorted(os.listdir(ori_path)):
    if i.startswith("._"):
        continue
    frames.append(transforms.ToTensor()(cv2.cvtColor(cv2.imread(os.path.join(ori_path, i)), cv2.COLOR_BGR2RGB)).unsqueeze(0))
OUT_DIR = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent"
with open(os.path.join(OUT_DIR, f"{vid}.STARTFRAME")) as f:
    start_frame = int(f.read())
    frames=frames[start_frame:start_frame+41]

ori_vid=torch.cat(frames, dim=0)
print(res_vid.shape, ori_vid.shape)
assert res_vid.shape==ori_vid.shape, "load video failed\n"
metric.update(res_vid, ori_vid)
print(metric.compute())