import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset.dataset import KITTIDataset, KITTIEvalDataset, HAOMOEvalDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_train_collate_fn(dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        """
            data_list: the data structure follows the return of getitem
            deal with the data in batch
            it will return the data to ——> enumerate(self.dataloader)
        """
        # print("call collate function")
        # Constructs a batch object
        pc1 = [e[0] for e in data_list]
        pc2 = [e[1] for e in data_list]
        scene_flow = [e[2] for e in data_list]
        # residual_flow = [e[4] for e in data_list]
        # instances = [e[4] for e in data_list]
        # transforms = [e[4] for e in data_list]
        batch_pc1 = torch.stack(pc1, dim=0)       # Produces (batch_size, n_points, 3) tensor
        batch_pc2 = torch.stack(pc2, dim=0)       # Produces (batch_size, n_points, 3) tensor
        batch_scene_flow = torch.stack(scene_flow, dim=0)       # Produces (batch_size, n_points, 3) tensor
        # batch_residual_flow = torch.stack(residual_flow, dim=0)       # Produces (batch_size, n_points) tensor
        # batch_instances = torch.stack(instances, dim=0)       # Produces (batch_size, n_points) tensor
        # batch_transforms = torch.stack(transforms, dim=0)       # Produces (batch_size, 4, 4) tensor

        return batch_pc1,batch_pc2, batch_scene_flow
        # return batch_pc1, batch_pc2, batch_mos

    return collate_fn


def make_eval_collate_fn(dataset):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        """
            data_list: the data structure follows the return of getitem
            deal with the data in batch
            it will return the data to ——> enumerate(self.dataloader)
        """
        # print("call collate function")
        # Constructs a batch object
        pc1 = [e[0] for e in data_list]
        pc2 = [e[1] for e in data_list]
        scene_flow = [e[2] for e in data_list]
        # residual_flow = [e[4] for e in data_list]
        # instances = [e[4] for e in data_list]
        # transforms = [e[4] for e in data_list]
        batch_pc1 = torch.stack(pc1, dim=0)       # Produces (batch_size, n_points, 3) tensor
        batch_pc2 = torch.stack(pc2, dim=0)       # Produces (batch_size, n_points, 3) tensor
        batch_scene_flow = torch.stack(scene_flow, dim=0)       # Produces (batch_size, n_points, 3) tensor
        # batch_residual_flow = torch.stack(residual_flow, dim=0)       # Produces (batch_size, n_points) tensor
        # batch_instances = torch.stack(instances, dim=0)       # Produces (batch_size, n_points) tensor
        # batch_transforms = torch.stack(transforms, dim=0)       # Produces (batch_size, 4, 4) tensor

        return batch_pc1,batch_pc2, batch_scene_flow
        # return batch_pc1, batch_pc2, batch_mos
    return collate_fn


def make_train_dataloader(config):
    dataset = KITTIDataset(config)
    # Reproducibility
    g = torch.Generator()
    g.manual_seed(42)

    train_collate_fn = make_train_collate_fn(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, collate_fn=train_collate_fn,
                            num_workers=config.num_workers, pin_memory=True, drop_last=True, shuffle=True,
                            worker_init_fn = seed_worker, generator = g)
    return dataloader


def make_eval_dataloader(config):
    test_dataset = KITTIEvalDataset(config)

    test_collate_fn = make_eval_collate_fn(test_dataset)
    dataloader = DataLoader(dataset=test_dataset, batch_size=config.eval_batch_size, collate_fn=test_collate_fn, shuffle=False)
    return dataloader

def make_haomo_eval_dataloader(config):
    test_dataset = HAOMOEvalDataset(config)

    test_collate_fn = make_eval_collate_fn(test_dataset)
    dataloader = DataLoader(dataset=test_dataset, batch_size=config.eval_batch_size, collate_fn=test_collate_fn, shuffle=False)
    return dataloader