import sys
sys.path.append('../../')
import torch
from rec_pangu.dataset import read_graph
from rec_pangu.models.ranking import lightgcn
from rec_pangu.trainer import GraphTrainer



if __name__ == '__main__':

    graph_path = 'sample_data/ratings.csv'
    train_rate = 0.7
    train_data, test_data,graph_information = read_graph(graph_path,train_rate)

    model = lightgcn(num_nodes=graph_information['num_nodes'], embedding_dim=32, num_layers=10)
    trainer = GraphTrainer()
    trainer.fit(model, train_data, epoch=10, lr=0.01)
    trainer.save_model(model, './model_ckpt')
    trainer.evaluate_model(model,test_data)

