import pdb
import sys
from CUB.train import parse_arguments
from CUB.train import (
    train_X_to_Proto_to_Y,
    train_X_to_C,
    train_oracle_C_to_y_and_test_on_Chat,
    train_Chat_to_y_and_test_on_Chat,
    train_X_to_C_to_y,
    train_X_to_y,
    train_X_to_Cy,
    # train_X_to_y_with_aux_C,
    train_probe,
    test_time_intervention,
    robustness,
    hyperparameter_optimization,
)

def run_experiments(dataset, args):
    experiment = args.exp

    if experiment == "Concept_XtoC":
        train_X_to_C(args)

    elif experiment == "APN":
        train_X_to_Proto_to_Y(args)

    elif experiment == "Independent_CtoY":
        train_oracle_C_to_y_and_test_on_Chat(args)

    elif experiment == "Sequential_CtoY":
        train_Chat_to_y_and_test_on_Chat(args)

    elif experiment == "Joint":
        train_X_to_C_to_y(args)

    elif experiment == "Standard":
        train_X_to_y(args)

    # todo
    # elif experiment == "StandardWithAuxC":
    #     train_X_to_y_with_aux_C(args)

    elif experiment == "Multitask":
        train_X_to_Cy(args)

    elif experiment == "Probe":
        train_probe(args)

    elif experiment == "TTI":
        test_time_intervention(args)

    elif experiment == "Robustness":
        robustness(args)

    elif experiment == "HyperparameterSearch":
        hyperparameter_optimization(args)



def parse_exp_arguments():
    """Helper function to read the dataset and Model type"""

    print(sys.argv)

    #! reading the dataset is currently not supported but might be
    #! a good idea for SUB etc.
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, "You need to specify dataset and experiment"
    assert sys.argv[1].upper() in ["OAI", "CUB"], "Please specify OAI or CUB dataset"
    assert sys.argv[2] in [
        "Concept_XtoC",
        "Independent_CtoY",
        "Sequential_CtoY",
        "Standard",
        "StandardWithAuxC",
        "Multitask",
        "Joint",
        "Probe",
        "TTI",
        "Robustness",
        "HyperparameterSearch",
        "APN",
    ], "Please specify valid experiment. Current: %s" % sys.argv[2]
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    # Handle accordingly to dataset
    args = parse_arguments(experiment=experiment)
    return dataset, args


if __name__ == "__main__":
    import torch
    import numpy as np

    dataset, args = parse_exp_arguments()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_experiments(dataset, args)
