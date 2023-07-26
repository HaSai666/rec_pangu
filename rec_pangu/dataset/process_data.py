# -*- ecoding: utf-8 -*-
# @ModuleName: process_data
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from .base_dataset import BaseDataset
from .multi_task_dataset import MultiTaskDataset
from .sequence_dataset import SequenceDataset, SequenceDatasetV2
import torch.utils.data as D


def get_base_dataloader(train_df, valid_df, test_df, schema, batch_size=512 * 3):
    train_dataset = BaseDataset(schema, train_df)
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = BaseDataset(schema, valid_df, enc_dict=enc_dict)
    test_dataset = BaseDataset(schema, test_df, enc_dict=enc_dict)

    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, valid_loader, test_loader, enc_dict


def get_multi_task_dataloader(train_df, valid_df, test_df, schema, batch_size=512 * 3):
    train_dataset = MultiTaskDataset(schema, train_df)
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = MultiTaskDataset(schema, valid_df, enc_dict=enc_dict)
    test_dataset = MultiTaskDataset(schema, test_df, enc_dict=enc_dict)

    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, valid_loader, test_loader, enc_dict


def get_sequence_dataloader(train_df, valid_df, test_df, schema, batch_size=512 * 3):
    train_dataset = SequenceDataset(schema, df=train_df, phase='train')
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = SequenceDataset(schema, df=valid_df, enc_dict=enc_dict, phase='test')
    test_dataset = SequenceDataset(schema, df=test_df, enc_dict=enc_dict, phase='test')

    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True, drop_last=True)

    return train_loader, valid_loader, test_loader, enc_dict

def get_sequence_dataloader_v2(df, schema, batch_size=512 * 3):
    train_dataset = SequenceDatasetV2(schema, df=df, phase='train')
    enc_dict = train_dataset.get_enc_dict()
    valid_dataset = SequenceDatasetV2(schema, df=df, enc_dict=enc_dict, phase='valid')
    test_dataset = SequenceDatasetV2(schema, df=df, enc_dict=enc_dict, phase='test')

    train_loader = D.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    valid_loader = D.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=True)
    test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=True, drop_last=True)

    return train_loader, valid_loader, test_loader, enc_dict


def get_dataloader(train_df, valid_df, test_df, schema, batch_size=512 * 3):
    if schema['task_type'] == 'ranking':
        return get_base_dataloader(train_df, valid_df, test_df, schema, batch_size=batch_size)
    elif schema['task_type'] == 'multitask':
        return get_multi_task_dataloader(train_df, valid_df, test_df, schema, batch_size=batch_size)
    elif schema['task_type'] == 'sequence':
        return get_sequence_dataloader(train_df, valid_df, test_df, schema, batch_size=batch_size)
    else:
        raise Exception(f"""task_type:{schema['task_type']} must be in ['ranking','multitask','sequence']""")


def get_single_dataloader(test_df, schema, enc_dict, batch_size=512, num_workers=0):
    if isinstance(schema['label_col'], list):
        test_dataset = MultiTaskDataset(schema, test_df, enc_dict=enc_dict)
        test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                   pin_memory=True)
        return test_loader
    else:
        test_dataset = BaseDataset(schema, test_df, enc_dict=enc_dict)
        test_loader = D.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                   pin_memory=True)
        return test_loader
