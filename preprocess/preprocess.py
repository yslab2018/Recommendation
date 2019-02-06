import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import time
import datetime
import random

ITEM_LOWER_LIMIT = 0 # 対象とするアイテムへのユーザ利用回数の下限
USER_LOWER_LIMIT = 0 # 対象とするアイテムのアイテム利用回数の下限

# 評価データに含める投稿日時の下限 (擬似データの期間は2018/01/01:00:00:00~2018/01/31:29:59:59)
d = [2018, 1, 15, 0, 0, 0]
d = datetime.datetime(d[0],d[1],d[2],d[3],d[4],d[5])
TIME_LIMIT = int(time.mktime(d.timetuple()))

# ユーザ履歴において考慮する系列長の最大値
SEQUENCE_LIMIT = 100

# 検証用データの層数
NUM_DEV = 10

def restrict_target(df, target_column, count_column, threshold):
    """
    各(ユーザ|アイテム)について，そのカウント対象(アイテム|ユーザ)の
    異なる利用を数え上げる
    """
    
    diff_count = {}
    
    pbar = tqdm(total=len(df))
    pbar.set_description('{} '.format(target_column))
    
    for i, row in df.iterrows():
        
        target_id = row[target_column]
        count_id  = row[count_column]    
        
        if target_id not in diff_count:
            diff_count[target_id] = [count_id]
        
        elif (target_id in diff_count) and (count_id not in diff_count[target_id]):
            diff_count[target_id].append(count_id)

        pbar.update(1)

    pbar.close()
            

    target = [] # 推薦対象となるターゲット(ユーザ|アイテム)を保存
    
    for num_id, count_list in diff_count.items():

        # カウント対象の数が閾値以上なら保存
        if len(count_list) >= threshold:
            target.append(num_id)
        
    return target


def main():
    
    df = pd.read_csv('../data/user_history.csv', sep=',', dtype={0:np.int32,2:np.int32,2:np.int64})

    #=========================
    #  Item-User extraction
    #=========================
    print('-----------------------------------------')
    print('Extraction of ItemID and UserID ...')
    
    # 閾値以下のアイテムを除外
    target_item = restrict_target(df, 'ItemID', 'UserID', ITEM_LOWER_LIMIT)
    tmp_df = df[df['ItemID'].isin(target_item)]
    
    # 閾値以下のユーザを除外
    target_user = restrict_target(tmp_df, 'UserID', 'ItemID', USER_LOWER_LIMIT)
    tmp_df = tmp_df[tmp_df['UserID'].isin(target_user)]

    
    #====================
    #    Data split
    #====================
    
    # train data
    tmp_train_term = df[df['Timestamp'] < TIME_LIMIT]
    
    users_freq = tmp_train_term['UserID'].value_counts()
    users_freq = users_freq[users_freq >= 2]
    tmp_users  = [i for i, v in  users_freq.iteritems()]

    train_term_df = tmp_train_term[tmp_train_term['UserID'].isin(tmp_users)]
    
    target_users = train_term_df['UserID'].unique()
    
    items_freq   = tmp_train_term['ItemID'].value_counts()
    target_items = [i for i, v in  items_freq.iteritems()]
    
    
    # eval data
    tmp_eval_term = df[df['Timestamp'] >= TIME_LIMIT]

    # 推薦対象以外のホテル・ユーザを除去
    eval_term_df = tmp_eval_term[(tmp_eval_term['ItemID'].isin(target_items)) & \
                                 (tmp_eval_term['UserID'].isin(target_users))]

    
    
    #=========================
    # Create training input 
    #=========================
    print('-----------------------------------------')
    print('Create input data ...')
    
    train_user_dict = {}
    
    pbar = tqdm(total=len(train_term_df))
    pbar.set_description('Train (1/2) ')

    for i, row in train_term_df.iterrows():
        user_id = row[0]
        item_id = row[1]
        time    = row[2]
        
        info_list = [time, item_id]
        
        if user_id not in train_user_dict:
            train_user_dict[user_id] = []
        
        train_user_dict[user_id].append(info_list)  
        pbar.update(1)
        
    pbar.close()

    
    # train sequence data
    train_data = []

    pbar = tqdm(total=len(train_user_dict))
    pbar.set_description('Train (2/2) ')
    
    for user_id, user_seq in train_user_dict.items():
        sort_user_seq = sorted(user_seq, key=lambda x:x[0], reverse=True)

        occur_item_list = {}
        
        n_seq = len(sort_user_seq)
        for i in range(n_seq-1):
            item_id = sort_user_seq[i][1]
            
            if  item_id not in occur_item_list:
                occur_item_list[item_id] = 1
                tmp = [row[1] for row in sort_user_seq[i:(i+SEQUENCE_LIMIT)]]
                tmp.reverse()                

                if len(tmp) >= 2:              
                    train_data.append(tmp)    
        
        pbar.update(1)
        del occur_item_list

    pbar.close()


    # shuhffle
    random.shuffle(train_data)
    
    with open('../input/train_data.pkl','wb') as fw:
        pickle.dump(train_data[NUM_DEV:], fw)

    with open('../input/dev_data.pkl','wb') as fw:
        pickle.dump(train_data[:NUM_DEV], fw)
    
    #=========================
    # Create training input 
    #=========================
    
    eval_user_dict = {}
    
    pbar = tqdm(total=len(eval_term_df))
    pbar.set_description('Eval  (1/2) ')
    
    for i, row in eval_term_df.iterrows():
        user_id = row[0]
        item_id = row[1]
        time    = row[2]
        
        info_list = [time, item_id]
        
        if user_id not in eval_user_dict:
            eval_user_dict[user_id] = []
        
        eval_user_dict[user_id].append(info_list)  
        pbar.update(1)

    
    pbar.close()
    
    # eval sequence data
    eval_data = []

    pbar = tqdm(total=len(eval_user_dict))
    pbar.set_description('Eval  (2/2) ')
    
    for user_id, user_seq in eval_user_dict.items():
        sort_user_seq = sorted(user_seq, key=lambda x:x[0], reverse=True)
        
        n_seq = len(sort_user_seq)
        for i in range(n_seq-1):
            length = len(sort_user_seq[i:])-1
            tmp = train_user_dict[user_id]
            tmp = sorted(tmp, key=lambda x:x[1])
            train_user_seq = [row[1] for row in tmp[-(SEQUENCE_LIMIT-length):]]
            
            eval_user_seq  = train_user_seq + [row[1] for row in sort_user_seq[i:]]
            eval_data.append(eval_user_seq)
        
        pbar.update(1)

    pbar.close()

    with open('../input/eval_data.pkl','wb') as fw:
        pickle.dump(eval_data, fw)
        
    
    print('\n=================================')
    print('Target/All Item : {0}/{1}'.format(len(target_items), len(df['ItemID'].unique()) ))
    print('Target/All User : {0}/{1}'.format(len(target_users), len(df['UserID'].unique()) ))
    print('=================================')
    print('Train sequence  = {0}'.format(len(train_data[NUM_DEV:])))
    print('Dev   sequence  = {0}'.format(NUM_DEV))
    print('Eval  sequence  = {0}'.format(len(eval_data)))
    print('=================================')
    
if __name__ == '__main__':
    main()
