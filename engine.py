from tqdm import tqdm
import torch.nn as nn
import torch
import config
# pos_weight=torch.tensor(3.2).to('cuda')
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.2).to('cuda'))(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), desc="Trainings"):
        ids = d['ids']
        mask = d['mask']
        target = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        
        if 'roberta' in config.MODEL_PATH:
            outputs = model(
                ids=ids, 
                mask=mask
            )

        else:
            token_type_ids = d['token_type_ids']
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            outputs = model(
                ids=ids, 
                token_type_ids=token_type_ids, 
                mask=mask
            )

        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device, test=False):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluation"):
            ids = d['ids']
            mask = d['mask']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)


            if test == False:
                target = d['targets']
                target = target.to(device, dtype=torch.float)

            if 'roberta' in config.MODEL_PATH:
                outputs = model(
                    ids=ids, 
                    mask=mask
                )
            else:
                token_type_ids = d['token_type_ids']
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                outputs = model(
                    ids=ids, 
                    token_type_ids=token_type_ids, 
                    mask=mask
                )

            if test == False:
                fin_targets.extend(target.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets
