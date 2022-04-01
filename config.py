### Config

import transformers

DEVICE = 'cuda'
EPOCHS = 6
LR = 3e-5
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
ACCUMULATION_STEPS = 4
MAX_LENGTH = 50
ADD_FEATURES = False  # if true change max_length to 60 
# MODEL_PATH = 'bert-base-uncased'
# MODEL_PATH = 'ai4bharat/indic-bert'
MODEL_PATH = 'xlm-roberta-base'
# MODEL_PATH = 'bert-base-multilingual-cased'
TRANSFORMER_CACHE = '/data/bashwani/.cache'

TRAIN_DATASET = './data/train_df.csv'
TEST_DATASET = './data/test_df.csv'

TOKENIZER = transformers.AutoTokenizer.from_pretrained(
  MODEL_PATH, 
  do_lower_case=True,
  cache_dir=TRANSFORMER_CACHE
)