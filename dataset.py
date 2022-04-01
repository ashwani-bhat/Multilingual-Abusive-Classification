### Dataset 
import config
import torch

class DatasetTraining:

  # init will be called first when we make the instance of the dataset
  def __init__(self, commentText, language=None, report_count_comment=None, report_count_post=None, like_count_comment=None, like_count_post=None, label=None):
    self.commentText  = commentText 
    self.language = language
    self.report_count_comment = report_count_comment
    self.report_count_post = report_count_post
    self.like_count_comment = like_count_comment
    self.like_count_post = like_count_post
    self.label = label
    self.max_len = config.MAX_LENGTH
    self.tokenizer = config.TOKENIZER

  # returns the length of the question
  def __len__(self):
    return len(self.commentText)
  
  # getter method
  def __getitem__(self, item):
    commentText = str(self.commentText[item])
    commentText = " ".join(commentText.split()) # split by all the whitespaces and join by one space

    # commentText += ". Comment language is " + self.language[item]
    secondText = None
    if config.ADD_FEATURES: 
      secondText = " reported "+str(self.report_count_comment[item])+" times and liked "+str(self.like_count_comment[item])+" times, language is "+ self.language[item] + "."
    
    
    # print(commentText)
    inputs = config.TOKENIZER.encode_plus(
      commentText,
      secondText, 
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation=True
    )

    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    
    if 'roberta' in config.MODEL_PATH:
      # return a dictionary 
      if self.label is None:
        return {
          'ids': torch.tensor(ids, dtype=torch.long),
          'mask': torch.tensor(mask, dtype=torch.long),
        }

      return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'targets': torch.tensor(int(self.label[item]), dtype=torch.float),
      }
    
    
    token_type_ids = inputs['token_type_ids']
    
    # return a dictionary 
    if self.label is None:
      return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
      }

    return {
      'ids': torch.tensor(ids, dtype=torch.long),
      'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
      'mask': torch.tensor(mask, dtype=torch.long),
      'targets': torch.tensor(int(self.label[item]), dtype=torch.float),
    }
    # for cross entropy use torch.long in dtype