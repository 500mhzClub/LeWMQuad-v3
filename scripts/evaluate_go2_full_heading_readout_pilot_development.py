"""Compare fixed continued heads on all 35 existing exposed pilot windows.

Reuses the completed pilot evaluator on CPU while native navigation uses GPU.
No fitting, new environments, or checkpoint selection is performed here.
"""
from scripts import evaluate_go2_correlation_motion_readout_development as evaluation
from scripts import train_go2_full_heading_readout_development as training


def main():
    heads = dict(original=training.original.load())
    heads.update({arm: training.load(arm) for arm in training.ARMS})
    paths = dict(original=training.original.OUTPUT/'readout.pt')
    paths.update({arm: training.OUTPUT/f'{arm}_final.pt' for arm in training.ARMS})
    evaluation.main(heads=heads, output=training.OUTPUT/'pilot_evaluation',
        head_metadata={arm: training.original.digest(path) for arm, path in paths.items()})


if __name__ == '__main__':
    main()
