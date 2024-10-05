import torch
from torch import nn

import torch.nn.utils.prune as prune


def train_model(trainer, model, data, retrain):
    # Train the model ⚡🚅⚡
    if retrain:
        data.stage = 1

    trainer.fit(model, data)
    return trainer


def print_global_sparsity(model_tups):
    print(
        'Global sparsity: {:.2f}%'.format(
            100. * float(sum(torch.sum(module.weight == 0) for module, _ in model_tups))
            / float(sum(module.weight.numel() for module, _ in model_tups))
        )
    )


def build_model_tups(model):
    module_tups = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module_tups.append((module, 'weight'))

    return module_tups


def prune_model_global_unstructured(model, proportion):
    module_tups = build_model_tups(model)
    print_global_sparsity(module_tups)

    prune.global_unstructured(
        parameters=module_tups,
        pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')

    module_tups = build_model_tups(model)
    print_global_sparsity(module_tups)

    return model
