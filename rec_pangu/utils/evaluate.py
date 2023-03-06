# -*- ecoding: utf-8 -*-
# @ModuleName: evaluate
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/12/21 22:13
import math
import numpy as np
import faiss
from sklearn.preprocessing import normalize

def get_recall_predict(model, test_data, device, topN=20,):
    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm='l2').astype('float32')
    hidden_size = item_embs.shape[1]
    faiss_index = faiss.IndexFlatIP(hidden_size)
    faiss_index.add(item_embs)

    preds = dict()
    for data in test_data:
        for key in data.keys():
            if key == 'user':
                continue
            data[key] = data[key].to(device)

        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        model.eval()
        user_embs = model(data, is_training=False)['user_emb']
        user_embs = user_embs.cpu().detach().numpy().astype('float32')

        user_list = data['user'].detach().cpu().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i,user in enumerate(user_list):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                preds[str(user)] = I[i, :]
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            for i,user in enumerate(user_list):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                item_list_set = []
                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序，内积越大，向量越近
                for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                preds[str(user)] = item_list_set
    return preds

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
