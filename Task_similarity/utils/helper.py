import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def smooth(y, window):
    box = np.ones(window)/window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_one_epoch(train_loss, model, 
                    loss_func, optimizer, 
                    replay_buffer, batch_size, 
                    model_type, device):
    data = replay_buffer.sample_batch(batch_size=batch_size)
    # preprocess data to concat state and action
    batch, label = preprocess_data(data, model_type=model_type)
    # batch, label = batch.to(device), label.to(device)
    optimizer.zero_grad()
    pred = model(batch)
    loss = loss_func(label, pred)
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    return train_loss


def eval_one_epoch(eval_loss, model, 
                    loss_func, eval_buffer, 
                    batch_size, 
                    model_type, device):
    eval_buffer.reset()
    temp_loss = 0
    while eval_buffer.iterate:
        eval_data = eval_buffer.sample_batch(batch_size=batch_size)
        eval_batch, eval_label = preprocess_data(eval_data, model_type=model_type)
        # eval_batch, eval_label = eval_batch.to(device), eval_label.to(device)
        with torch.no_grad(): 
            eval_pred = model(eval_batch)
        iter_loss = loss_func(eval_label, eval_pred)
        temp_loss += iter_loss.item()    
    eval_loss.append(temp_loss/eval_buffer.batch_num)  

    return eval_loss


def preprocess_data(data, model_type='dynamics'):
    obs = data['obs']
    act = data['act']
    rew = data['rew']
    obs2 = data['obs2']
    done = data['done']

    # concat state and action as raw input to the model
    raw_input = torch.cat((obs, act), 1)  # convert into shape: (batch_size, state_dim+act_dim)
    if model_type == 'dynamics':
        label = obs2
    elif model_type == 'reward':
        label = rew
        label = label.unsqueeze(1)  # conbert shape: (batch) --> (batch, 1)
    else:
        raise Exception('Model type is not correct: must be dynamics or reward')

    return raw_input, label


def update_model(model_type, model, replay_buffer,
                eval_buffer, optimizer, 
                loss_func, batch_size,
                device): 

    # early stopping 
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    
    # torch.manual_seed(random_seed)
    # model.to(device)
    # train model
    model.train() 
    train_loss = []
    eval_loss = []
    epoch = 0 
    while True:
        epoch += 1
        train_loss = train_one_epoch(train_loss, model, loss_func, 
                                    optimizer, replay_buffer, 
                                    batch_size, model_type, device)

        eval_loss = eval_one_epoch(eval_loss, model, 
                    loss_func, eval_buffer, 
                    batch_size, 
                    model_type, device)

        # early stopping
        latest_validation_loss = eval_loss[-1]
        if early_stopper.early_stop(latest_validation_loss): 
            break
        
    return train_loss, eval_loss



