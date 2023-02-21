# -*- ecoding: utf-8 -*-
# @ModuleName: evaluate
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/12/21 22:13
import math

def evaluate_recall(preds,test_gd, topN=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f'recall@{topN}': round(recall,4), f'ndcg@{topN}': round(ndcg,4), f'hitrate@{topN}': round(hitrate,4)}
