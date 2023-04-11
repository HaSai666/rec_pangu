# -*- ecoding: utf-8 -*-
# @ModuleName: ranking_server
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/4/11 09:35
import onnx
from onnx_tf.backend import prepare
import torch
import os


def construct_demmy_data(schema: dict) -> tuple:
    """
    Construct dummy data for the model input.

    Args:
    schema (dict): A dictionary containing the schema of the input data.

    Returns:
    tuple: A tuple containing the dummy input and dynamic axes.
    """
    dummy_input = {}
    dynamic_axes = {}
    # Iterate through dense columns and create dummy data
    for col in schema['dense_cols']:
        dummy_input.update({col: torch.rand((8))})
        dynamic_axes.update({col: [0]})
    # Iterate through sparse columns and create dummy data
    for col in schema['sparse_cols']:
        dummy_input.update({col: torch.randint(0, 1, (8, 1)).squeeze().long()})
        dynamic_axes.update({col: [0]})
    return dummy_input, dynamic_axes


def export2tf(model: torch.nn.Module, schema: dict, serving_dir: str, version: str) -> None:
    """
    Export the PyTorch model to TensorFlow format.

    Args:
    model (torch.nn.Module): The PyTorch model to be exported.
    schema (dict): A dictionary containing the schema of the input data.
    serving_dir (str): The directory where the exported model will be saved.
    version (str): The version number of the exported model.
    """
    os.makedirs(serving_dir, exist_ok=True, mode=0o777)
    # Define the path to save the ONNX model
    onnx_path = os.path.join(serving_dir, 'temp.onnx')
    # Construct dummy data for the model input
    dummy_input, dynamic_axes = construct_demmy_data(schema)
    # Export the PyTorch model to ONNX format
    torch.onnx.export(model, (dummy_input, False),
                      onnx_path,
                      input_names=schema['dense_cols'] + schema['sparse_cols'],
                      output_names=['pred'],
                      dynamic_axes=dynamic_axes)
    # Load the ONNX model
    model_onnx = onnx.load(onnx_path)
    # Convert the ONNX model to TensorFlow format
    tf_rep = prepare(model_onnx)
    # Export the TensorFlow model
    tf_rep.export_graph(os.path.join(serving_dir, str(version)))
