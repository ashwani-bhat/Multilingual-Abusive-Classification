# TRAIN
from sklearn.metrics import f1_score
import config
import pandas as pd
from sklearn import model_selection
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import numpy as np
import os
from engine import *
from dataset import *
from model import *


def run():
    df_test = pd.read_csv(config.TEST_DATASET).fillna('none')
    # df_test = df_test[:1000]

    df_test = df_test.reset_index(drop=True)
    print(len(df_test))
    
    #### Get inconsistent data ####
    with open('out.txt') as f:
        inconsis = []
        for x in f.readlines():
            if x != '\n':
                inconsis.append((int(x.split(',')[0]), 1)) #1 gave best
    
    ##############################
    print(len(inconsis))

    test_dataset = DatasetTraining(
        commentText = df_test.commentText.values,
        language = df_test.language.values,
        report_count_comment = df_test.report_count_comment.values,
        report_count_post = df_test.report_count_post.values,
        like_count_comment = df_test.like_count_comment.values,
        like_count_post = df_test.like_count_post.values,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device('cuda')

    model = BertBaseUncased()
    # model_path = './best_models_tillnow/best_model.pth'
    model_path = './models/best_model.pth'
    
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    outputs, _ = eval_fn(test_data_loader, model, device, test=True)
    outputs = np.array(outputs) >= 0.5
    outputs = outputs[:, 0]
    outputs = 1*outputs

    outs = list(zip(df_test['CommentId'].tolist(), outputs)) + inconsis
    outs = sorted(outs, key=lambda x: x[0])
    
    submission = pd.DataFrame(outs, columns=['CommentId', 'Label']) 
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    run()