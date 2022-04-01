# TRAIN
from sklearn.metrics import f1_score
import config
import pandas as pd
from sklearn import model_selection
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import numpy as np
import random
import os
from engine import *
from dataset import *
from model import *

# TODO: use weighted loss
def run():
    dfx = pd.read_csv(config.TRAIN_DATASET).fillna('none')
    # dfx = dfx[:5000]
    df_train, df_valid = model_selection.train_test_split(
    dfx, 
    test_size=0.15,
    random_state=42,
    stratify=dfx.label.values   # stratify ensures that training and validation, has same ratio of positive and negative samples
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    print(df_train['label'].value_counts())
    print(df_valid['label'].value_counts())

    train_dataset = DatasetTraining(
      commentText = df_train.commentText.values,
      language = df_train.language.values,
      report_count_comment = df_train.report_count_comment.values,
      report_count_post = df_train.report_count_post.values,
      like_count_comment = df_train.like_count_comment.values,
      like_count_post = df_train.like_count_post.values,
      label = df_train.label.values  
    )

    train_data_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = config.TRAIN_BATCH_SIZE,
    num_workers=2
    )

    valid_dataset = DatasetTraining(
      commentText = df_valid.commentText.values,
      language = df_train.language.values,
      report_count_comment = df_train.report_count_comment.values,
      report_count_post = df_train.report_count_post.values,
      like_count_comment = df_train.like_count_comment.values,
      like_count_post = df_train.like_count_post.values,
      label = df_valid.label.values  
    )

    valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size = config.VALID_BATCH_SIZE,
    num_workers=1
    )

    device = torch.device('cuda')

    model = BertBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
    {
      'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
      'weight_decay': 0.001
    },
    {
      'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
      'weight_decay': 0.0
    }
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model) # for multiple GPUs

    best_accuracy = 0
    best_f1 = 0
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, model, device)

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1 = f1_score(targets, outputs)

        print(f"Accuracy score = {accuracy}")
        print(f"F1 score = {f1}")
        print(f"Epoch {epoch} done. . .")

        if f1 > best_f1:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), './models/epoch'+str(epoch)+'.pth')
            torch.save(model.state_dict(), './models/best_model.pth')
            print("Best model Saved. . .")
            best_accuracy = accuracy
            best_f1 = f1


if __name__ == '__main__':
    """Sets random seed everywhere."""
    print("Seed set")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    run()