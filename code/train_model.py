# -*- coding: utf-8 -*-
# Author : Stefan
# Time : 2021-09-23
# Desc : 模型训练

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
import pickle 
from sklearn.metrics import f1_score
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
from attack import FGM, PGD
import io, sys
sys.path.append('..')
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
from dataProcess.dataEnhance import eda_data
from utils.Logger import initlog


CFG = { #训练的参数配置
    'fold_num': 5, # 五折交叉验证
    'seed': 2,
    'lr': 3e-5, #学习率
    'model': './save_model/pretrain_model_parameter/chinese_roberta_wwm_ext_pytorch',  # 网络不好用这个
    # 'model': 'hfl/chinese-roberta-wwm-ext',  # 也可以直接从huggingface下载
    'max_len': 512, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 10, #batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'num_workers': 0,
    'accum_iter': 1, #梯度累积，相当于将batch_size*2
    'weight_decay': 2e-4, #权重衰减，防止过拟合
    'device': 0,
    'model_index': 'q',
    'is_enhance': True,
    'sub_label': True,
    'add_stage1_train': True,
    'attack_mode': 'fgm',
    'split_science_law': 0, # 是否把数据划分为文理科；0表示不做划分，1表示取理科数据，2表示取文科数据
    'is_choose_layer': False,
    'layers': [0, -1],  # [-4, -3, -2, -1], 
    'pooling': 'mean',
    'eda_repeat_n': 5,
}    

log_path = 'train.log' 
logger = initlog(logfile= "./logs/" + log_path)
logger.info('pid:{}'.format(os.getpid()))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_train(model_index, is_enhance=False, sub_label=False, add_stage1_train=False):
    CFG['model_index'] = model_index
    CFG['is_enhance'] = is_enhance
    CFG['sub_label'] = sub_label
    CFG['add_stage1_train'] = add_stage1_train

    if os.path.exists('./datasets/tmp_data/label_id2cat.pkl'):
        exist_label_id2cat = pickle.load(open('./datasets/tmp_data/label_id2cat.pkl', 'rb'))
    else:
        exist_label_id2cat = {}

    if CFG['model_index'].startswith('k'):
        train_df = pd.read_csv("./datasets/tmp_data/{}_train_data.csv".format(CFG['model_index']))
        train_df = train_df.sample(frac=1)
        print('len of train data:{}'.format(len(train_df)))
        train_df = train_df[['type', 'Content', 'q_Level', 'Analysis', 'options', 'TestQuestionID', 'k_Level', 'KnowledgeID']]
        train_df['Analysis'] = train_df['Analysis'].fillna('')
        num_labels = len(train_df['KnowledgeID'].unique())

        if '{}_label_id2cate'.format(CFG['model_index']) in exist_label_id2cat:
            label_id2cate = exist_label_id2cat['{}_label_id2cate'.format(CFG['model_index'])]
        else:
            label_id2cate = dict(enumerate(train_df['KnowledgeID'].unique()))
            exist_label_id2cat['{}_label_id2cate'.format(CFG['model_index'])] = label_id2cate
            pickle.dump(exist_label_id2cat, open('./datasets/tmp_data/label_id2cat.pkl', 'wb'))

        label_cate2id = {value: key for key, value in label_id2cate.items()}
        train_df['label'] = train_df['KnowledgeID'].map(label_cate2id)


    elif CFG['model_index'].startswith('q'):
        train_df = pd.read_csv("./datasets/tmp_data/new_train_data.csv")

        # tmp for program
        if CFG['split_science_law'] == 1:
            train_df = train_df[train_df['TestQuestionID'] <= 4854]  
            SPLIT = '_science'
        elif CFG['split_science_law'] == 2:
            train_df = train_df[train_df['TestQuestionID'] > 4854]
            CFG['add_stage1_train'] = False
            SPLIT = '_law'
        else:
            SPLIT = ''

        train_df = train_df.drop_duplicates(subset='TestQuestionID', keep='first').reset_index()
        if CFG['sub_label']:
            train_df = train_df[train_df['q_Level'].isin([1,2,3,4,5])].reset_index()


        print('len of train data:{}'.format(len(train_df)))
        train_df = train_df[['type', 'Content', 'q_Level', 'Analysis', 'options', 'TestQuestionID', 'k_Level', 'KnowledgeID']]
        train_df['Analysis'] = train_df['Analysis'].fillna('')
        num_labels = len(train_df['q_Level'].unique())

        if '{}{}_label_id2cate'.format(CFG['model_index'], SPLIT) in exist_label_id2cat:
            label_id2cate = exist_label_id2cat['{}{}_label_id2cate'.format(CFG['model_index'], SPLIT)]
        else:
            label_id2cate = dict(enumerate(train_df['q_Level'].unique()))
            exist_label_id2cat['{}{}_label_id2cate'.format(CFG['model_index'], SPLIT)] = label_id2cate
            pickle.dump(exist_label_id2cat, open(f'./datasets/{SUB}label_id2cat.pkl', 'wb'))
        label_cate2id = {value: key for key, value in label_id2cate.items()}
        train_df['label'] = train_df['q_Level'].map(label_cate2id)

        # add stage1 train data
        if CFG['add_stage1_train']:
            stage1_train_df = pd.read_csv("./datasets/tmp_data/sub_train_stage1_data.csv")
            stage1_train_df = stage1_train_df.drop_duplicates(subset='TestQuestionID', keep='first').reset_index()
            stage1_train_df = stage1_train_df[['type', 'Content', 'q_Level', 'Analysis', 'options', 'TestQuestionID', 'k_Level', 'KnowledgeID']]
            stage1_train_df['Analysis'] = stage1_train_df['Analysis'].fillna('')
            stage1_train_df['label'] = stage1_train_df['q_Level'].map(label_cate2id)


    class MyDataset(Dataset):
        def __init__(self, dataframe):
            self.df = dataframe

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):

            label = self.df.label.values[idx]
            type = self.df.type.values[idx].replace('\r', '').replace('\n', '')
            content = self.df.Content.values[idx].replace('\r', '').replace('\n', '')
            analysis = self.df.Analysis.values[idx].replace('\r', '').replace('\n', '')
            options = self.df.options.values[idx].replace('\r', '').replace('\n', '')
            question = (type + '[SEP]' + content + '[SEP]' + analysis)
            return question, options, label

    def collate_fn(data):
        input_ids, attention_mask, token_type_ids, label = [], [], [], []
        for x in data:
            text = tokenizer(x[0], text_pair=x[1], padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
            input_ids.append(text['input_ids'].squeeze().tolist())
            attention_mask.append(text['attention_mask'].squeeze().tolist())
            token_type_ids.append(text['token_type_ids'].squeeze().tolist())
            label.append(x[-1])
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)
        label = torch.tensor(label)
        return input_ids, attention_mask, token_type_ids, label

    class AverageMeter:  # 为了tqdm实时显示loss和acc
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


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


    def train_model(model, fgm,pgd,train_loader):  # 训练一个epoch
        model.train()

        losses = AverageMeter()
        accs = AverageMeter()

        optimizer.zero_grad()

        tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

        for step, (input_ids, attention_mask, token_type_ids, label) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), label.to(device).long()

            with autocast():  # 使用半精度训练

                if CFG['is_choose_layer']:
                    output = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(output, y) / CFG['accum_iter']
                else:
                    output = model(input_ids, attention_mask, token_type_ids)[0]
                    loss = criterion(output, y) / CFG['accum_iter']

                scaler.scale(loss).backward()

                if CFG['attack_mode'] == 'fgm':
                    fgm.attack()  # 在embedding上添加对抗扰动
                else:
                    pgd.attack()

                if CFG['is_choose_layer']:
                    output2 = model(input_ids, attention_mask, token_type_ids)
                    loss2 = criterion(output2, y) / CFG['accum_iter']
                else:
                    output2 = model(input_ids, attention_mask, token_type_ids)[0]
                    loss2 = criterion(output2, y)/ CFG['accum_iter']

                scaler.scale(loss2).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                if CFG['attack_mode'] == 'fgm':
                    fgm.restore() # 恢复 embedding 参数
                else:
                    pgd.restore()

                if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

            acc = (output.argmax(1) == y).sum().item() / y.size(0)
            losses.update(loss.item() * CFG['accum_iter'], y.size(0))
            accs.update(acc, y.size(0))
            tk.set_postfix(loss=losses.avg, acc=accs.avg)

        return losses.avg, accs.avg


    def test_model(model, val_loader):  # 验证
        model.eval()

        losses = AverageMeter()
        accs = AverageMeter()
        y_truth, y_pred = [], []

        with torch.no_grad():
            tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            for idx, (input_ids, attention_mask, token_type_ids, label) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), label.to(device).long()

                if CFG['is_choose_layer']:
                    output = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(output, y) / CFG['accum_iter']
                else:
                    output = model(input_ids, attention_mask, token_type_ids).logits

                y_truth.extend(y.cpu().numpy())
                y_pred.extend(output.argmax(1).cpu().numpy())
                loss = criterion(output, y)
                acc = (output.argmax(1) == y).sum().item() / y.size(0)
                losses.update(loss.item(), y.size(0))
                accs.update(acc, y.size(0))

                tk.set_postfix(loss=losses.avg, acc=accs.avg)

        micro_f1 = f1_score(y_truth, y_pred, average="micro")
        print('evaluate micro_f1:{}'.format(round(micro_f1, 4)))
        return losses.avg, accs.avg, micro_f1


    tokenizer = BertTokenizer.from_pretrained(CFG['model'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed'])\
                        .split(np.arange(train_df.shape[0]), train_df.label.values) #五折交叉验证

    for fold, (trn_idx, val_idx) in enumerate(folds):
        train = train_df.loc[trn_idx]
        val = train_df.loc[val_idx]

        train_set = MyDataset(train)
        val_set = MyDataset(val)

        train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                                num_workers=CFG['num_workers'])
        val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                                num_workers=CFG['num_workers'])

        if CFG['is_choose_layer']:
            model = SelfNet(CFG['model'], num_labels=num_labels).to(device) 
        else:
            model = BertForSequenceClassification.from_pretrained(CFG['model'],num_labels=num_labels).to(device)  # 模型

        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
        criterion = nn.CrossEntropyLoss()
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                    CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
        # get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
        fgm = FGM(model)
        pgd = PGD(model)
        save_path = './save_model/{}_learning/'.format(CFG['model_index'])

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        best_f1 = 0.0
        for epoch in range(CFG['epochs']):
            if CFG['is_enhance']:
                train_eda = eda_data(train, repeat_n=CFG['eda_repeat_n'])
                if CFG['add_stage1_train']:
                    train_eda = pd.concat([train_eda, stage1_train_df])
                train_set = MyDataset(train_eda)
            elif CFG['add_stage1_train']:
                train_addstage1 = pd.concat([train, stage1_train_df])
                train_set = MyDataset(train_addstage1)


            val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                                    num_workers=CFG['num_workers'])
            train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                                    num_workers=CFG['num_workers'])
            time.sleep(0.2)

            train_loss, train_acc = train_model(model, fgm, pgd, train_loader)
            val_loss, val_acc, F1 = test_model(model, val_loader)
            logger.info(f'fold:{fold}, epoch:{epoch}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_f1:{F1:.4f}')
            if F1 > best_f1 and epoch >= 3:
                best_f1 = F1
                save_model_path = save_path + '5fold_{}_{}_fgm_{}.pt'.format(fold,epoch,round(F1,3))
                if CFG['split_science_law'] > 0:
                    prefix = 'science' if CFG['split_science_law'] == 1 else 'law'
                    save_model_path = save_path + '{}_5fold_{}_{}_fgm_{}.pt'.format(prefix,fold,epoch,round(F1,3))
                torch.save(model.state_dict(), save_model_path)


def main():
    for model_index in ['k1', 'k2', 'k3', 'k4', 'k5', 'q']:
        sub_label = True if model_index == 'q' else False
        is_enhance = True if model_index in ['k5', 'q'] else False
        add_stage1_train = True if model_index == 'q' else False
        model_train(model_index, is_enhance, sub_label, add_stage1_train)


if __name__ == '__main__':
    main()