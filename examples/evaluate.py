import warnings

import torch
from aux_bn import MixBatchNorm2d

warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def load_multi(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    state_dict = {k[7:]: v for k, v in state_dict['state_dict'].items()}
    return state_dict

def run_evaluation():
    models = [
        "resnet50m",
        "resnet50d",
        "resnet50s",
        "resnet50t"
    ]
    state_dicts = [
        lambda: torch.load('/mnt/c/Projects/imagenetx/merged_model_checkpoint.pth.tar')['state_dict'],
        lambda: load_multi('/mnt/c/Projects/imagenetx/res50-debiased.pth.tar'),
        lambda: load_multi('/mnt/c/Projects/imagenetx/res50-shape-biased.pth.tar'),
        lambda: load_multi('/mnt/c/Projects/imagenetx/res50-texture-biased.pth.tar'),
    ]
    assert len(models)==len(state_dicts)
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {
        "batch_size": 64,
        "print_predictions": True,
        "num_workers": 2,

        "model_args": {
            "norm_layer": MixBatchNorm2d
        },


        #####################################
        "state_dicts": state_dicts
    }
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
