import torch
from torcheval.metrics import PeakSignalNoiseRatio
from torchmetrics.functional.image import structural_similarity_index_measure
import torchvision.transforms as transforms
import os
import cv2

metric = PeakSignalNoiseRatio()
vid="98_zPGqEMMWyx4"
ori_path=f"/home/twtomtwcc00/VSPW/data/{vid}/origin"
res_path="./deblur_test_out.mp4"
OUT_DIR = "/home/twtomtwcc00/VideoBlurRemoval/VSPW_latent"


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

with open(os.path.join(OUT_DIR, f"{vid}.STARTFRAME")) as f:
    start_frame = int(f.read())

for i in sorted(os.listdir(ori_path)):
    if i.startswith("._"):
        continue
    frame=cv2.imread(os.path.join(ori_path, i))
    frame=cv2.resize(frame, (1536, 864))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transforms.ToTensor()(frame).unsqueeze(0)  # (1, 3, H, W)
    frames.append(frame)
frames=frames[start_frame:start_frame+41]

ori_vid=torch.cat(frames, dim=0)
print(res_vid.shape, ori_vid.shape)
assert res_vid.shape==ori_vid.shape, "load video failed\n"
metric.update(res_vid, ori_vid)
print("PSNR:"+str(metric.compute()))
print(structural_similarity_index_measure(res_vid, ori_vid))
