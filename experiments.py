import random
import pdb
import sys
import argparse


def run_experiments(dataset, experiment, args):

    if dataset == 'OAI':
        from OAI.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_y_with_aux_C,
            train_X_to_Cy,
            train_probe,
            test_time_intervention,
            hyperparameter_optimization,
        )

    elif dataset == 'CUB':
        from CUB.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            #train_excluding_each_attribute,  # Remove each concept and training
            train_concept_level,
            train_data_level,
            train_single_level,
            #train_test,
            run_eval,
            #train_excluding_random_attributes,  # Randomly remove 10 concept and training
            test_time_intervention,
            robustness,
            hyperparameter_optimization,
        )

    elif dataset == 'CelebA':
        from CelebA.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            #train_excluding_each_attribute,  # Remove each concept and training
            train_concept_level,
            train_data_level,
            train_single_level,
            #train_test,
            run_eval,
            #train_excluding_random_attributes,  # Randomly remove 10 concept and training
            test_time_intervention,
            robustness,
            hyperparameter_optimization,
        )

    elif dataset == 'Chexpert':
        from Chexpert.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            #train_excluding_each_attribute,  # Remove each concept and training
            train_concept_level,
            train_data_level,
            train_single_level,
            #train_test,
            run_eval,
            #train_excluding_random_attributes,  # Randomly remove 10 concept and training
            test_time_intervention,
            robustness,
            hyperparameter_optimization,
        )

    #experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Exclude_Concept_One_By_One':
        train_excluding_each_attribute(args[0])
    elif experiment == 'DataLevel':
        train_data_level(args[0])
    elif experiment == 'ConceptLevel':
        train_concept_level(args[0])
    elif experiment == 'Eval':
        run_eval(args[0])
    elif experiment == 'SingleLevel':
        train_single_level(args[0])    
    elif experiment == 'Test':
        train_test(args[0])


    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'StandardWithAuxC':
        train_X_to_y_with_aux_C(*args)

    elif experiment == 'Multitask':
        train_X_to_Cy(*args)

    elif experiment == 'Probe':
        train_probe(*args)

    elif experiment == 'TTI':
        test_time_intervention(*args)

    elif experiment == 'Robustness':
        robustness(*args)

    elif experiment == 'HyperparameterSearch':
        hyperparameter_optimization(*args)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['OAI', 'CUB', 'CelebA', 'Chexpert'])
    parser.add_argument(
        'experiment',
        type=str,
        choices={
            'Concept_XtoC',
            'Independent_CtoY',
            'Sequential_CtoY',
            'Standard',
            'StandardWithAuxC',
            'Multitask',
            'Joint',
            'Probe',
            'TTI',
            'Robustness',
            'HyperparameterSearch',
            'Exclude_Concept_One_By_One',
            'Exclude_Random_Concepts',
            'DataLevel',
            'ConceptLevel',
            'SingleLevel',
            'Test',
            'Eval',
        },
    )
    parser.add_argument('seed', type=int)
    args = parser.parse_args(sys.argv[1:4])
    dataset, experiment, seed = args.dataset, args.experiment, args.seed

    if dataset == 'OAI':
        from OAI.train import parse_arguments as parse_dataset_args
    elif dataset == 'CUB':
        from CUB.train import parse_arguments as parse_dataset_args
    elif dataset == 'CelebA':
        from CelebA.train import parse_arguments as parse_dataset_args
    elif dataset == 'Chexpert':
        from Chexpert.train import parse_arguments as parse_dataset_args
    
    args = parse_dataset_args(experiment, seed, sys.argv[4:])
    return dataset, experiment, seed, args


if __name__ == '__main__':

    import torch
    import numpy as np

    dataset, experiment, seed, args = parse_arguments()

    # Seeds
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.zeros(1).device == 'cpu': print('[WARNING] You are using CPU device')

    run_experiments(dataset, experiment, args)
