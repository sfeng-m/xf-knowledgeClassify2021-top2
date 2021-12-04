# -*- coding: utf-8 -*-
# Author : Stefan
# Time : 2021-09-23
# Desc : 将多个子模型的结果进行合并

import pickle
import pandas as pd
# import know_map

result_dict = {}
for k in ['k1', 'k2', 'k3', 'k4', 'k5', 'q']:
    id_result = pickle.load(open(f'./result/sub_result/{k}_result.pkl', 'rb'))
    # id_result = pickle.load(open(f'./result/sub_result/tmp4test_v2_baseline_{k}_result.pkl', 'rb'))
    result_dict.update(id_result)

test_df = pd.read_csv('./datasets/test_data.csv')
test_df['KnowledgeID'] = test_df[['TestQuestionID', 'k_Level']].apply(lambda x: result_dict[str(x[0])+x[1]], axis=1)
# test_df = know_map.main(test_df)  # 这部分代码由队友完成，所以无法开源，敬请谅解。
test_df['q_Level'] = test_df[['TestQuestionID', 'q_Level']].apply(lambda x: 
                            int(x[1]) if x[1]==-1 else int(result_dict[str(int(x[0]))]), axis=1)

test_df[['index', 'TestQuestionID', 'KnowledgeID', 'q_Level']].to_csv('./result/result.csv',index=None)
