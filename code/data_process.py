# -*- coding: utf-8 -*-
# Author : Stefan
# Time : 2021-09-23
# Desc : 数据预处理操作，包括清洗脏数据，数据切分，构造伪标签数据等
import pandas as pd
import copy
import Levenshtein
import io, sys
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


def clean_data():
    """清洗脏数据
    """
    train_df = pd.read_csv('./datasets/train_data.csv')

    # 清洗同一TestQuestionID同一k_Level同一KnowledgeID条件下存在不同q_Level的数据
    df_dup = train_df.groupby(['TestQuestionID', 'k_Level', 'KnowledgeID'])['q_Level'].unique().to_frame().reset_index()
    df_dup['len'] = df_dup['q_Level'].apply(lambda x: len(x))
    df_dirty = df_dup.loc[df_dup['len'] > 1]
    for qid, klevel, knowid in df_dirty[['TestQuestionID', 'k_Level', 'KnowledgeID']].values:
        train_df = train_df.drop(train_df[(train_df['TestQuestionID']==qid) & (train_df['k_Level']==klevel) & (train_df['KnowledgeID']==knowid)].index)
    
    # 清洗同一TestQuestionID同一k_Level条件下存在不同KnowledgeID的数据
    df_dup1 = train_df.groupby(['TestQuestionID', 'k_Level'])['KnowledgeID'].unique().to_frame().reset_index()
    df_dup1['len'] = df_dup1['KnowledgeID'].apply(lambda x: len(x))
    df_dirty = df_dup1.loc[df_dup1['len'] > 1]
    for qid, klevel in df_dirty[['TestQuestionID', 'k_Level']].values:
        train_df = train_df.drop(train_df[(train_df['TestQuestionID']==qid) & (train_df['k_Level']==klevel)].index)
        
    train_df.to_csv('./datasets/tmp_data/new_train_data.csv', index=False)


def split_kdata():
    """将清洗后的数据按知识点类别进行划分，方便对各类别进行数据探索
    """
    train_df = pd.read_csv('./datasets/tmp_data/new_train_data.csv')
    train_df = train_df[['type', 'Content', 'q_Level', 'Analysis', 'options', 'TestQuestionID', 'k_Level', 'KnowledgeID']]
    train_df['Analysis'] = train_df['Analysis'].fillna('')
    train_k1 = train_df[train_df['k_Level']=='CD']
    train_k2 = train_df[train_df['k_Level']=='CR']
    train_k3 = train_df[train_df['k_Level']=='JE']
    train_k4 = train_df[train_df['k_Level']=='TS']
    train_k5 = train_df[train_df['k_Level']=='KL']

    train_k1.to_csv('./datasets/tmp_data/k1_train_data.csv', index=False)
    train_k2.to_csv('./datasets/tmp_data/k2_train_data.csv', index=False)
    train_k3.to_csv('./datasets/tmp_data/k3_train_data.csv', index=False)
    train_k4.to_csv('./datasets/tmp_data/k4_train_data.csv', index=False)
    train_k5.to_csv('./datasets/tmp_data/k5_train_data.csv', index=False)


def get_stage1_diff_pseudo_label(edit_ratio=0.1):
    """根据编辑距离获取相似数据并以对应的标签作为伪标签数据参与训练
    """
    train_stage1 = pd.read_csv('./datasets/stage1_train_data.csv')
    test_stage2 = pd.read_csv('./datasets/test_data.csv')
    train_stage1 = train_stage1.drop_duplicates(subset='TestQuestionID', keep='first')
    test_stage2 = test_stage2.drop_duplicates(subset='TestQuestionID', keep='first')
    stage1_value = train_stage1[['Content', 'Analysis', 'TestQuestionID']].values
    stage1_dict = {'##'.join([str(value[0]), str(value[1])]): value[2] for value in stage1_value}
    stage2_value = test_stage2[['Content', 'Analysis', 'TestQuestionID']].values
    stage2_dict = {'##'.join([str(value[0]), str(value[1])]): value[2] for value in stage2_value}
    remain_ids = []
    for stage2_content, stage2_qid in stage2_dict.items():
        if stage2_content in stage1_dict:
            remain_ids.append(stage1_dict[stage2_content])
    for stage2_content, stage2_qid in stage2_dict.items():
        for stage1_content, stage1_qid in stage1_dict.items():
            edit = Levenshtein.distance(stage2_content, stage1_content) / max(len(stage1_content), len(stage2_content))
            if edit <= edit_ratio and stage1_qid not in remain_ids:
                remain_ids.append(stage1_qid)
    sub_train_stage1 = train_stage1[train_stage1.TestQuestionID.isin(remain_ids)]
    # sub_test_stage2 = test_stage2[train_stage1.TestQuestionID.isin(remain_ids)]
    sub_train_stage1.to_csv('./datasets/tmp_data/sub_train_stage1_data.csv', index=False)


def main():
    clean_data()
    split_kdata()
    get_stage1_diff_pseudo_label(edit_ratio=0.1)
    

if __name__ == '__main__':
    main()
