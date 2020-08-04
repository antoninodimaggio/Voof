import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

def put_speed_on_video(mp4_path, pred_text_path, act_text_path):
    pred_speed_list = np.around(np.loadtxt(pred_text_path), decimals=1)
    act_speed_list = np.around(np.loadtxt(act_text_path), decimals=1)[1:]
    video = cv2.VideoCapture(mp4_path)
    video.set(1, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.VideoWriter('./docs/demos/readme_media/promo.mp4', 0x7634706d, 20, (640, 480))
    for t in count():
        ret, frame = video.read()
        if ret == False or t >= len(pred_speed_list):
            break
        pred_curr_speed = pred_speed_list[t]
        act_curr_speed = act_speed_list[t]
        cv2.putText(frame,
            f'Speed: {pred_curr_speed}',
            (50, 50),
            font,
            0.7,
            (51, 100, 51),
            2,
            cv2.LINE_4)
        cv2.putText(frame,
            f'Error: {round(pred_curr_speed - act_curr_speed, 1)}',
            (50, 80),
            font,
            0.7,
            (82, 51, 255),
            2,
            cv2.LINE_4)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()


def parse_logs(log_file_path):
    train_loss = []
    val_loss = []
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.replace("\'", "\"")
            line_dict = json.loads(line)
            train_loss.append(line_dict['train_epoch_loss'])
            val_loss.append(line_dict['eval_epoch_loss'])
    return train_loss, val_loss


def graph_loss(log_file_path_farn, log_file_path_pwc):
    farn_train_loss, farn_val_loss = parse_logs(log_file_path_farn)
    pwc_train_loss, pwc_val_loss = parse_logs(log_file_path_pwc)
    with plt.style.context('seaborn-muted'):
        _, ax = plt.subplots(figsize=(20,6))
        ax.plot(range(1, len(farn_train_loss)+1), farn_train_loss, alpha=0.7, linewidth=3, label='Farneback Train Loss')
        ax.plot(range(1, len(farn_train_loss)+1), farn_val_loss, alpha=0.7, linewidth=3, label='Farneback Eval Loss')
        ax.plot(range(1, len(pwc_train_loss)+1), pwc_train_loss, alpha=0.7, linewidth=3, label='PWC Train Loss')
        ax.plot(range(1, len(pwc_train_loss)+1), pwc_val_loss, alpha=0.7, linewidth=3, label='PWC Eval Loss')
        ax.set_xticks(range(1, len(pwc_train_loss)+1))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        plt.savefig('./docs/readme_media/loss.png')


if __name__ == '__main__':
    graph_loss('./training_logs/farneback.log', './training_logs/pwc.log')
