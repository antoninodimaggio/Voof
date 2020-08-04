import logging
import torch
from cnn.NVIDIA import NVIDIA
from train_eval_utils import generate_optical_flow_dataloader, generate_optical_flow_dataloader_split


# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def train(flow_train_dataset, flow_eval_dataset, num_channels, save_model_path, log_name, epochs):
    logger = logging.getLogger('training')
    logger.addHandler(logging.FileHandler(f'./training_logs/{log_name}.log', mode='w'))
    device = torch.device('cuda')
    model = NVIDIA(num_channels).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
        patience=0, threshold=0.2, threshold_mode='abs', min_lr=1e-8,)
    print('Start training ...')
    for epoch in range(epochs):
        model.train()
        training_loss = 0
        eval_loss = 0
        for i, (flow_stack, speed_vector) in enumerate(flow_train_dataset):
            flow_stack, speed_vector = flow_stack.to(device), speed_vector.to(device)
            optimizer.zero_grad()
            pred_speed = model(flow_stack)
            loss = criterion(pred_speed, speed_vector)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        model.eval()
        for i, (flow_stack, speed_vector) in enumerate(flow_eval_dataset):
            flow_stack, speed_vector = flow_stack.to(device), speed_vector.to(device)
            with torch.no_grad():
                pred_speed = model(flow_stack)
                loss = criterion(pred_speed, speed_vector)
                eval_loss += loss.item()

        logging_dict = {"epoch": epoch+1,
                        "train_epoch_loss": training_loss/len(flow_train_dataset),
                        "eval_epoch_loss": eval_loss/len(flow_eval_dataset),
                        "lr": optimizer.param_groups[0]['lr']}
        logger.info('%s', logging_dict)
        scheduler.step(training_loss/len(flow_train_dataset))
    print('Training complete!')
    torch.save(model.state_dict(), save_model_path)
    print('Model saved!')


def evaluate_and_write(flow_eval_dataset, num_channels, model_path, save_text_path):
    all_pred_speeds = []
    device = torch.device('cuda')
    criterion = torch.nn.MSELoss()
    model = NVIDIA(num_channels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    eval_loss = 0
    print('Start evaluation ...')
    for i, (flow_stack, speed_vector) in enumerate(flow_eval_dataset):
        flow_stack, speed_vector = flow_stack.to(device), speed_vector.to(device)
        with torch.no_grad():
            pred_speed = model(flow_stack)
            loss = criterion(pred_speed, speed_vector)
            eval_loss += loss.item()
            all_pred_speeds.append(pred_speed.item())
    total_eval_loss = eval_loss/len(flow_eval_dataset)
    print(f'The total MSE eval loss is: {total_eval_loss}')
    write_to_file(all_pred_speeds, save_text_path)
    print('Done evaluating!')


def write_to_file(data, path):
    with open(path, 'w') as txt_file:
        for ele in data:
            txt_file.write(f'{ele}\n')


if __name__ == '__main__':
    # flow_train_dataset, flow_eval_dataset = generate_optical_flow_dataloader_split('/freespace/local/ajd311/farnebeck_train.pt', 0.2, train_batch_size=32)
    # train(flow_train_dataset, flow_eval_dataset, 3, './trained_models/nvidia/farneback_model.pt', 'farneback', 20)
    # evaluate_and_write(flow_eval_dataset, 3, './trained_models/nvidia/farneback_model.pt', './docs/demos/farneback.txt')

    flow_train_dataset, flow_eval_dataset = generate_optical_flow_dataloader_split('/freespace/local/ajd311/pwc_train.pt', 0.2, train_batch_size=32)
    train(flow_train_dataset, flow_eval_dataset, 2, './trained_models/nvidia/pwc_model.pt', 'pwc', 26)
