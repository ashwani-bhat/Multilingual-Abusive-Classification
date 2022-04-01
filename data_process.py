test_csv =  './data/ShareChat-IndoML-Datathon-NSFW-CommentChallenge_Test_NoLabel.csv'
train_csv = './data/ShareChat-IndoML-Datathon-NSFW-CommentChallenge_Train.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import re
import emoji
import string 

def clean_comments(comments):
    cleaned = comments   
    cleaned = emoji.demojize(cleaned, delimiters=("", " "))
    # cleaned = re.sub(r'[^A-Za-z0-9 _]+', '', cleaned).replace('_', ' ')
    cleaned = re.sub(',(?!(?=[^"]*"[^"]*(?:"[^"]*"[^"]*)*$))', '', cleaned).replace('_', ' ')
    cleaned = re.sub(r'\d', " mobile number ", cleaned)
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))

    cleaned = cleaned.split()
    cleaned = list(dict.fromkeys(cleaned))
    cleaned = " ".join(cleaned)
    return cleaned


def create_train_df(c):    # removed inconsistent and waste data
    with open(c) as file:
        data = file.readlines()
    
    headers = data[0].split(',') 
    headers[-1] = headers[-1][:-1]

    count = 0
#     data = data[:20000]
    v = []
    inconsistent = []
    for id_no, d in enumerate(tqdm(data[1:], total=len(data[1:]))):
        values = d.split(',')
        values[-1] = values[-1][:-1]
        if len(values) != len(headers):
            count += 1
            try:
                idx, comment, restofthings = d.split('"')
                idx = re.sub(r'[^0-9]+', '', idx)
                values = [idx, comment] + restofthings[1:].split(',')   # idx has comma at the end, restofthings has empty space at start and \n for last element
                values[-1] = values[-1][:-1]
            except:
                inconsistent.append(d)
                continue
            
        values[1] = clean_comments(values[1])   # these are comments, need to clean this as much as I can
        v.append(values)
    
    df = pd.DataFrame(v, columns=headers)
    
    df = df.astype(dtype={"CommentId":int, "commentText":str, "language": str, 'user_index': int, 'post_index': int, 'report_count_comment': int, 'report_count_post': int, 'like_count_comment': int, 'like_count_post': int, 'label': int})

    return df, inconsistent

def create_test_df(c):   # kept waste data row so that it matches the submission number
    with open(c) as file:
        data = file.readlines()
    
    headers = data[0].split(',') 
    headers[-1] = headers[-1][:-1]

    count = 0
    v = []
    inconsistent = []
    for id_no, d in enumerate(tqdm(data[1:], total=len(data[1:]))):
        values = d.split(',')
        values[-1] = values[-1][:-1]
        if len(values) != len(headers):
            count += 1
            try:
                idx, comment, restofthings = d.split('"')
                idx = re.sub(r'[^0-9]+', '', idx)
                values = [idx, comment] + restofthings[1:].split(',')   # idx has comma at the end, restofthings has empty space at start and \n for last element
                values[-1] = values[-1][:-1]
            except:
                print(d)
                inconsistent.append(d)
                continue
            
        values[1] = clean_comments(values[1])   # these are comments, need to clean this as much as I can
        v.append(values)
    
    df = pd.DataFrame(v, columns=headers)
    
    df = df.astype(dtype={"CommentId":int, "commentText":str, "language": str, 'user_index': int, 'post_index': int,
       'report_count_comment': int, 'report_count_post': int, 'like_count_comment': int,
       'like_count_post': int})
        
    return df, inconsistent

def test_create_kaggle():
    def read_csv(path:str):
        file = open(path, "r").read()
        ix = []
        ctx = []

        for i, row in enumerate(file.split("\n")):
            l = re.sub(',(?!(?=[^"]*"[^"]*(?:"[^"]*"[^"]*)*$))', "\t", row)
            try:
                lk = l.split("\t")
                if len(lk)>2 and len(lk[0])<6:
                    p,q= lk[0], lk[1]
                    ix.append(p)
                    ctx.append(q)
                else:
                    lk=row.replace('"', " ")
                    lk=lk.split(",")
                    p,q = lk[0], lk[1]
                    ix.append(p)
                    ctx.append(clean_comments(q))
            except Exception as e:
                print(i)
                print("Exception occurred!.", e)
                print(f"Length of ids obtained: {len(ix)}, and text: {len(ctx)}")

        df = pd.DataFrame()
        df["CommentId"]=ix[1:]
        df["commentText"]=ctx[1:]
        df = df.astype(dtype={"CommentId":int, "commentText":str})
        return df

    df = read_csv("./data/ShareChat-IndoML-Datathon-NSFW-CommentChallenge_Test_20_Percent_NoLabel.csv")
    return df

# df_test = test_create_kaggle()
# df_test.to_csv('./data/test_df_kaggle.csv', index=False)

# train_df, inconsistent = create_train_df(train_csv)
# print(len(train_df), len(inconsistent), print(train_df.dtypes))
# train_df.to_csv('./data/train_df.csv', index=False)

test_df, inconsistent = create_test_df(test_csv)
print(len(test_df), len(inconsistent), print(test_df.dtypes))

test_df.to_csv('./data/test_df.csv', index=False)
# breakpoint()