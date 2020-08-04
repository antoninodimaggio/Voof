import cv2
import numpy as np
import torch
import torchvision.transforms as T
from cnn.PWC import pwc_dc_net
from preprocess_farneback import (change_brightness, generate_frames)
from tqdm import tqdm

device = torch.device('cuda')
model = pwc_dc_net('./trained_models/pwc/pwc_net.pth.tar').to(device)
model.eval()


def transform(image, bright_factor):
    """augment brightness, crop/resize"""
    image = change_brightness(image, bright_factor)
    image = cv2.resize(image[100:440, :-90], (192, 64), interpolation = cv2.INTER_AREA)
    return T.ToTensor()(image)


def scale_flow_back(flow):
    """scale the flow back up"""
    flow = flow.squeeze(0).cpu().data.numpy()
    # multiply by 20 for now (PWC-Net divides by 20)
    flow = flow * 20
    u = flow[0]
    v = flow[1]
    u_ = T.ToTensor()(cv2.resize(u, (220, 66)))
    v_ = T.ToTensor()(cv2.resize(v, (220, 66)))
    flow_stack = torch.cat([u_, v_])
    return flow_stack.unsqueeze(0)


def pwc_calc_dense_optical_flow(prev_frame, curr_frame, bright_factor):
    prev_frame = transform(prev_frame, bright_factor).unsqueeze(0).to(device)
    curr_frame = transform(curr_frame, bright_factor).unsqueeze(0).to(device)
    with torch.no_grad():
        flow = model(prev_frame, curr_frame)
    scaled_flow = scale_flow_back(flow)
    return scaled_flow


def generate_optical_flow_dataset_pwc(mp4_path, text_path):
    """generate dataset from mp4 and txt"""
    for t, (prev_frame, curr_frame) in enumerate(tqdm(generate_frames(mp4_path), desc='Generating dense optical flow tensors')):
        bright_factor = 0.2 + np.random.uniform()
        flow_tensor = pwc_calc_dense_optical_flow(prev_frame, curr_frame, bright_factor)
        if t == 0:
            flow_stack = flow_tensor
        else:
            flow_stack = torch.cat([flow_stack, flow_tensor])
    # can't estimate speed of first frame
    speed_vector = np.loadtxt(text_path)[1:]
    flow_dataset = torch.utils.data.TensorDataset(flow_stack,
                                                  torch.from_numpy(speed_vector).float())
    return flow_dataset


def save_whole_set(mp4_path, text_path, save_path):
    """save whole dataset"""
    flow_dataset = generate_optical_flow_dataset_pwc(mp4_path, text_path)
    torch.save(flow_dataset, save_path)


if __name__ == '__main__':
    save_whole_set('./data/train/train.mp4', './data/train/train.txt', '/freespace/local/ajd311/pwc_train.pt')
