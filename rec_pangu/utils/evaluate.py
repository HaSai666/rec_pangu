# -*- ecoding: utf-8 -*-
# @ModuleName: evaluate
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/12/21 22:13
from typing import Dict, List
import math
import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize


def get_recall_predict(model: torch.nn.Module,
                       test_data: torch.utils.data.DataLoader,
                       device: torch.device,
                       topN: int = 20) -> dict:
    f"""
        Given the trained model, test data and device, get the recommendations
        for all users in the test data using Faiss index.

    Args:
        model: Trained model.
        test_data: Test data for evaluation.
        device: Device used for evaluation.
        topN: Number of recommendations to consider per user. Default is 20.

    Returns:
        Dictionary containing recommendations for all users.
    """

    # Get the item embeddings and add them to Faiss index.
    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm='l2').astype('float32')
    hidden_size = item_embs.shape[1]
    faiss_index = faiss.IndexFlatIP(hidden_size)
    faiss_index.add(item_embs)

    # Iterate through all users in the test data and get their recommendations.
    preds = dict()
    for data in test_data:
        for key in data.keys():
            if key == 'user':
                continue
            data[key] = data[key].to(device)

        # Get user embeddings for the given data.
        model.eval()
        user_embs = model(data, is_training=False)['user_emb']
        user_embs = user_embs.cpu().detach().numpy().astype('float32')

        user_list = data['user'].cpu().numpy()

        # Get the recommendations using Faiss index.

        if len(user_embs.shape) == 2:

            # Non-multi-interest model.
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)

            for i, user in enumerate(user_list):
                preds[str(user)] = I[i, :]

        else:

            # Multi-interest model.
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = faiss_index.search(user_embs, topN)
            for i, user in enumerate(user_list):
                item_list_set = []
                item_list = list(zip(np.reshape(I[i * ni:(i + 1) * ni], -1),
                                     np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                preds[str(user)] = item_list_set
    return preds


def evaluate_recall(preds: Dict[str, List[int]],
                    test_gd: Dict[str, List[int]],
                    topN: int = 50) -> Dict[str, float]:
    """ Calculates recall, ndcg, and hitrate.

    Args:
        preds: A dictionary of user_id and a list of predicted item_id.
        test_gd: A dictionary of user_id and a list of actual item_id.
        topN: An integer representing the top N predicted items.

    Returns:
        A dictionary containing the recall, ndcg, and hitrate.
    """

    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0

    # Iterate over each user in the test data
    for user in test_gd.keys():
        if user not in preds.keys():
            continue
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]

        # Iterate over each actual item in the test data
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                # Increment recall for each correctly predicted item
                recall += 1
                # Calculate dcg
                dcg += 1.0 / math.log(no + 2, 2)

            # Calculate idcg
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)

        # Calculate total recall, total ndcg, and total hitrate
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1

    # Calculate overall recall, overall ndcg, and overall hitrate
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total

    # Return a dictionary containing results
    return {f'recall@{topN}': round(recall, 4), f'ndcg@{topN}': round(ndcg, 4), f'hitrate@{topN}': round(hitrate, 4)}

# def get_recall_predict(model, test_data, device, topN=20,):
#     item_embs = model.output_items().cpu().detach().numpy()
#     item_embs = normalize(item_embs, norm='l2').astype('float32')
#     hidden_size = item_embs.shape[1]
#     faiss_index = faiss.IndexFlatIP(hidden_size)
#     faiss_index.add(item_embs)
#
#     preds = dict()
#     for data in test_data:
#         for key in data.keys():
#             if key == 'user':
#                 continue
#             data[key] = data[key].to(device)
#
#         # 获取用户嵌入
#         # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
#         # 其他模型，shape=(batch_size, embedding_dim)
#         model.eval()
#         user_embs = model(data, is_training=False)['user_emb']
#         user_embs = user_embs.cpu().detach().numpy().astype('float32')
#
#         user_list = data['user'].detach().cpu().numpy()
#
#         # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
#         if len(user_embs.shape) == 2:  # 非多兴趣模型评估
#             user_embs = normalize(user_embs, norm='l2').astype('float32')
#             D, I = faiss_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
#             for i,user in enumerate(user_list):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
#                 preds[str(user)] = I[i, :]
#         else:  # 多兴趣模型评估
#             ni = user_embs.shape[1]  # num_interest
#             user_embs = np.reshape(user_embs,
#                                    [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
#             user_embs = normalize(user_embs, norm='l2').astype('float32')
#             D, I = faiss_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
#             for i,user in enumerate(user_list):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
#                 item_list_set = []
#                 # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
#                 item_list = list(
#                     zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
#                 item_list.sort(key=lambda x: x[1], reverse=True)  # 降序排序，内积越大，向量越近
#                 for j in range(len(item_list)):  # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
#                     if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
#                         item_list_set.append(item_list[j][0])
#                         if len(item_list_set) >= topN:
#                             break
#                 preds[str(user)] = item_list_set
#     return preds

# def evaluate_recall(preds,test_gd, topN=50):
#     total_recall = 0.0
#     total_ndcg = 0.0
#     total_hitrate = 0
#     for user in test_gd.keys():
#         recall = 0
#         dcg = 0.0
#         item_list = test_gd[user]
#         for no, item_id in enumerate(item_list):
#             if item_id in preds[user][:topN]:
#                 recall += 1
#                 dcg += 1.0 / math.log(no+2, 2)
#             idcg = 0.0
#             for no in range(recall):
#                 idcg += 1.0 / math.log(no+2, 2)
#         total_recall += recall * 1.0 / len(item_list)
#         if recall > 0:
#             total_ndcg += dcg / idcg
#             total_hitrate += 1
#     total = len(test_gd)
#     recall = total_recall / total
#     ndcg = total_ndcg / total
#     hitrate = total_hitrate * 1.0 / total
#     return {f'recall@{topN}': round(recall,4), f'ndcg@{topN}': round(ndcg,4), f'hitrate@{topN}': round(hitrate,4)}
