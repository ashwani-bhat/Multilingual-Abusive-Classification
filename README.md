# Multilingual Abusive Comment Detection

The dataset for this project is taken from Kaggle: https://www.kaggle.com/competitions/multilingualabusivecomment

This dataset include:

- Massive & Multilungual: 10+ low-resource Indic languages
- Human annotated
- Metadata for each comment (Eg. #likes, #reports, etc.)

## Dataset Stats
Training dataset:
- No. of Non-abusive Comments: 1148019
- No. of Abusive Comments: 352879

## Directory Structure
    .
    ├── config.py                   # contains the hyperparameters and file path locations
    ├── dataset.py                  # dataset class for processing the dataloader
    ├── engine.py                   # training and evaluation epochs
    ├── model.py                    # model class
    └── train.py                    # script to perform model training
    └── eval.py                     # script to perform model evaluation
    └── data_process.py             # data preprocessing to remove noise, to generate relevant text from the data
    └── utils.py                    # helper functions
    
    
## Example run
For training
```bash
python train.py
```

Best model will be saved here: `./models/best_model.pth`. For any changes in the hyperparameters, do it in the config.py file.

For evaluation
```bash
python eval.py
```

Final predictions will be stored here: `submission.csv`
