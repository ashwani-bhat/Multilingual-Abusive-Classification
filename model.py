# MODEL CLASS
import torch
from torch import nn
import config
import transformers

''' 
This class defines your model architecture,
As an example for this:
> Bert + dropout(0.2) + output layer (Linear Layer)

'''

class BertBaseUncased(nn.Module):
    def __init__(self, add_cnn=False, dropout=0.1, kernel_num=3, kernel_sizes=[2,3,4], num_labels=1):
        super(BertBaseUncased, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(config.MODEL_PATH, cache_dir=config.TRANSFORMER_CACHE, output_hidden_states=True)
        self.bert = transformers.AutoModel.from_pretrained(config.MODEL_PATH, cache_dir=config.TRANSFORMER_CACHE, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.output1 = nn.Linear(768, 768) # first Linear(number of input features, number of outputs)
        self.activation = nn.ReLU()
        self.output2 = nn.Linear(768, 1) # second Linear
        
        self.add_cnn = add_cnn
        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc = nn.Linear(442, 1)
        self.flat = nn.Flatten()

    def forward(self, ids, mask, token_type_ids=None):
        ''' 
        Bert model returns two outputs o1, o1
        o1: sequence output | for every token, you will get a vector of hidden_length size
        o2: pooled output | one vector of hidden_length size
        o2 is the output of [CLS] token
        '''
        if token_type_ids == None:
            out = self.bert(input_ids=ids, attention_mask=mask)
        else:
            out = self.bert(input_ids=ids, token_type_ids=token_type_ids, attention_mask=mask)

        seq_out, pool_out, hidden_out = out[0], out[1], out[2]
        
        ### mean pool 
        # pool the last 4 [CLS] tokens hidden state
        h12 = hidden_out[-1][:, 0].reshape((-1, 1, self.config.hidden_size))
        h11 = hidden_out[-2][:, 0].reshape((-1, 1, self.config.hidden_size))
        h10 = hidden_out[-3][:, 0].reshape((-1, 1, self.config.hidden_size))
        h9  = hidden_out[-4][:, 0].reshape((-1, 1, self.config.hidden_size))
        all_hidden = torch.cat([h9, h10, h11, h12], dim=1)
        mean_pool = torch.mean(all_hidden, dim=1)

        #####
        # o2 = pool_out
        o2 = mean_pool

        if self.add_cnn:
            breakpoint()
            x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in hidden_out]), 0), 0, 1)
            x = self.pool(self.dropout(self.activation(self.conv(self.dropout(x)))))
            x = self.fc(self.dropout(self.flat(self.dropout(x))))
            return x

        o2 = self.dropout(o2)
        o2 = self.output1(o2)
        o2 = self.activation(o2)
        return self.output2(o2)
        ''' 
        In the above we can take o1 as well and then do the mean pooling,
        max pooling and then concatenate them
        '''