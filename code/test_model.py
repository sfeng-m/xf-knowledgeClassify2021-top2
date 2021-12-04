
# -*- coding: utf-8 -*-
# Author : Stefan
# Time : 2021-09-23
# Desc : 模型预测

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from sklearn.model_selection import *
from transformers import *

CFG = { #训练的参数配置
    'lr': 3e-5, #学习率
    'model': './save_model/pretrain_model_parameter/chinese_roberta_wwm_ext_pytorch',  # 网络不好用这个
    # 'model': 'hfl/chinese-roberta-wwm-ext',  # 也可以直接从huggingface下载
    'max_len': 512, #文本截断的最大长度
    'valid_bs': 16,
    'num_workers': 0,
    'device': 0,
    'model_index': 'k1',
    'sub_label': False,
    'split_science_law': 0,
    'is_choose_layer': False,
    'layers': [-4, -3, -2, -1], # [0, -1],  # 
    'pooling': 'max',
}

def model_predict(model_index, sub_label=False):
    CFG['model_index'] = model_index
    CFG['sub_label'] = sub_label

    tokenizer = BertTokenizer.from_pretrained(CFG['model'])
    torch.cuda.set_device(CFG['device'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k_level_map = {'k1': 'CD', 'k2': 'CR', 'k3': 'JE', 'k4':'TS', 'k5':'KL'}
    test_df = pd.read_csv('./datasets/test_data.csv')
    if CFG['model_index'].startswith('k'):
        test_df = test_df[test_df['k_Level']==k_level_map[CFG['model_index']]]
    else:
        test_df = test_df.drop_duplicates(subset='TestQuestionID', keep='first').reset_index()
        if CFG['split_science_law'] == 1:
            test_df = test_df[test_df['TestQuestionID'] <= 4872]
        elif CFG['split_science_law'] == 2:
            test_df = test_df[test_df['TestQuestionID'] > 4872]

    test_df['Analysis'] = test_df['Analysis'].fillna('')

    label_id2cate = pickle.load(open('./datasets/tmp_data/label_id2cat.pkl', 'rb'))
    num_labels = len(label_id2cate['{}_label_id2cate'.format(CFG['model_index'])])


    class MyDataset(Dataset):
        def __init__(self, dataframe):
            self.df = dataframe

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            type = self.df.type.values[idx]
            content = self.df.Content.values[idx]
            analysis = self.df.Analysis.values[idx]
            options = self.df.options.values[idx].replace('\r', '').replace('\n', '')
            question = (type + '[SEP]' + content + '[SEP]' + analysis).replace('\r', '').replace('\n', '')
            return question, options

    def collate_fn(data):
        input_ids, attention_mask, token_type_ids = [], [], []
        for x in data:
            text = tokenizer(x[0], text_pair=x[1], padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
            input_ids.append(text['input_ids'].squeeze().tolist())
            attention_mask.append(text['attention_mask'].squeeze().tolist())
            token_type_ids.append(text['token_type_ids'].squeeze().tolist())
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)
        return input_ids, attention_mask, token_type_ids

    class SelfNet(nn.Module):
        def __init__(self, model_name, num_labels):
            super(SelfNet,self).__init__()
            self.autoModel=BertModel.from_pretrained(model_name)
            self.classifier=nn.Linear(768, num_labels)
            self.dropout = nn.Dropout(0.1)

        def forward(self,input_ids, attention_mask, token_type_ids):
            outputs=self.autoModel(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
            encoded_layers = outputs['hidden_states']
            hidden_state = []
            for l in CFG['layers']:
                hidden_state.append(encoded_layers[l][:, 0].unsqueeze(1))
            hidden_state = torch.cat(hidden_state, dim=1)

            if CFG['pooling'] == 'max':
                hidden_state, _ = torch.max(hidden_state, dim=1)
            elif CFG['pooling'] == 'mean':
                hidden_state = torch.mean(hidden_state, dim=1)
            else:
                hidden_state = hidden_state.view(hidden_state.size(0), -1)

            hidden_state = self.dropout(hidden_state)
            output = self.classifier(hidden_state)
            return output


    test_df['label'] = 0
    test_set = MyDataset(test_df)
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=CFG['num_workers'])

    if CFG['is_choose_layer']:
        model = SelfNet(CFG['model'], num_labels=num_labels).to(device) 
    else:
        model = BertForSequenceClassification.from_pretrained(CFG['model'],num_labels=num_labels).cuda()  # 模型

    y_pred,predictions=[],[]
    y_all = np.zeros((len(test_df),num_labels))

    idx_model = []
    model_path = f"./save_model/{CFG['model_index']}_learning/"
    model_files = os.listdir(model_path)
    model_files = [file for file in model_files if file.startswith('5fold')]
    for fold in range(5):
        sub_model_files = [file for file in model_files if int(file.split('_')[1])==fold]
        idx_model.append(model_path + sorted(sub_model_files)[-1])   

    for m in idx_model:
        model.load_state_dict(torch.load(m, map_location='cuda:{}'.format(CFG['device'])))
        y_pred = []
        with torch.no_grad():
            tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
            for idx, (input_ids, attention_mask, token_type_ids) in enumerate(tk):
                input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device)

                if CFG['is_choose_layer']:
                    output = model(input_ids, attention_mask, token_type_ids)
                else:
                    output = model(input_ids, attention_mask, token_type_ids).logits
                y_pred.extend(output.cpu().numpy())

            y_all = y_all+np.array(y_pred)

    test_df['pred'] = y_all.argmax(1)

    SPLIT = ''
    if CFG['model_index'].startswith('k'):
        test_df['KnowledgeID'] = test_df['pred'].map(label_id2cate['{}_label_id2cate'.format(CFG['model_index'])])
        id_result = {}
        for TestQuestionID, k_Level, KnowledgeID in test_df[['TestQuestionID', 'k_Level', 'KnowledgeID']].values:
            id_result[str(TestQuestionID)+k_Level] = KnowledgeID
    else:
        if CFG['split_science_law'] == 1:
            SPLIT = '_science'
        elif CFG['split_science_law'] == 2:
            SPLIT = '_law'
        test_df['q_Level'] = test_df['pred'].map(label_id2cate['{}{}_label_id2cate'.format(CFG['model_index'], SPLIT)])
        id_result = {}
        for TestQuestionID, q_Level in test_df[['TestQuestionID', 'q_Level']].values:
            id_result[str(TestQuestionID)] = q_Level
    
    pickle.dump(id_result, open('./result/sub_result/{}_result.pkl'.format(CFG['model_index']), 'wb'))


def main():
    for model_index in ['k1', 'k2', 'k3', 'k4', 'k5', 'q']:
        sub_label = True if model_index == 'q' else False
        model_predict(model_index, sub_label)


if __name__ == '__main__':
    main()