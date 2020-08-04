import cv2
import numpy as np
import torch
import torchvision.transforms as T
from itertools import count
from tqdm import tqdm


def generate_frames(mp4_path):
    # (height, width, number_of_channels) = (480, 640, 3)
    video = cv2.VideoCapture(mp4_path)
    _, prev_frame = video.read()
    for t in count():
        ret, curr_frame = video.read()
        if ret == False:
            break
        yield prev_frame, curr_frame
        prev_frame = curr_frame
    video.release()
    cv2.destroyAllWindows()


def change_brightness(image, bright_factor):
    """augment brightness"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image_rgb


def transform(image, bright_factor):
    """augment brightness, crop/resize"""
    image = change_brightness(image, bright_factor)
    image = cv2.resize(image[100:440, :-90], (220, 66), interpolation = cv2.INTER_AREA)
    return image


def calc_dense_optical_flow(prev_frame, curr_frame, bright_factor):
    prev_frame, curr_frame = transform(prev_frame, bright_factor), transform(curr_frame, bright_factor)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[:,:,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                        None, 0.5, 1, 15, 2, 5, 1.3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow


def generate_optical_flow_dataset(mp4_path, text_path):
    """generate dataset from mp4 and txt"""
    for t, (prev_frame, curr_frame) in enumerate(tqdm(generate_frames(mp4_path), desc='Generating dense optical flow tensors')):
        bright_factor = 0.2 + np.random.uniform()
        rgb_flow = calc_dense_optical_flow(prev_frame, curr_frame, bright_factor)
        rgb_flow_tensor = T.ToTensor()(rgb_flow).unsqueeze(0)
        if t == 0:
            flow_stack = rgb_flow_tensor
        else:
            flow_stack = torch.cat([flow_stack, rgb_flow_tensor])
    # can't estimate speed of first frame
    speed_vector = np.loadtxt(text_path)[1:]
    flow_dataset = torch.utils.data.TensorDataset(flow_stack,
                                                  torch.from_numpy(speed_vector).float())
    return flow_dataset


def save_whole_set(mp4_path, text_path, save_path):
    """save whole dataset"""
    flow_dataset = generate_optical_flow_dataset(mp4_path, text_path)
    torch.save(flow_dataset, save_path)


if __name__ == '__main__':
    save_whole_set('./data/train/train.mp4', './data/train/train.txt', '/freespace/local/ajd311/farneback_train.pt')
