import optuna

from CUB.train import parse_arguments, train_X_to_Proto_to_Y

BASE_ARGS = ["--seed", "1", "-ckpt", "1" , "-e" ,"300", "-optimizer", "sgd", "-pretrained", "-use_aux" ,"-use_attr", "-weighted_loss", "multiple", "-data_dir", "../data/CUB_processed/class_attr_data_10", "-n_attributes", "312",  "-attr_loss_weight", "1", "-normalize_loss", "-b", "64", "-end2end"]
             

def objective(trial: optuna.Trial):
    # General but fixed arguments
    args = parse_arguments("Joint", BASE_ARGS)

    # General arguments to optimize
    args.lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001])  # type: ignore
    args.scheduler_step = trial.suggest_categorical( # type: ignore
        "scheduler_step", [10, 15, 20, 1000]
    )
    args.weight_decay = trial.suggest_categorical("weight_decay", [0.0004, 0.00004])  # type: ignore

    # ProtoMod specific properties
    args.proto_n_vectors = trial.suggest_int("proto_n_vectors", 1, 10)  # type: ignore
    args.proto_use_groups = trial.suggest_categorical("proto_use_groups", [True, False])  # type: ignore

    # Weighting of loss terms
    args.proto_weight_attribute_reg = trial.suggest_float( # type: ignore
        "proto_weight_attribute_reg", 1e-6, 1, log=True
    )  
    args.proto_weight_cpt = trial.suggest_float("proto_weight_cpt", 1e-9, 1e-2, log=True) # type: ignore
    args.proto_weight_decorrelation = trial.suggest_float("proto_weight_decorrelation", 1e-6, 1e-2) # type: ignore

    return train_X_to_Proto_to_Y(args)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best:", study.best_trial.params)
