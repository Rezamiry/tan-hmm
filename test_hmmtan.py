import random
import pandas as pd
import numpy as np

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from hmmtan import HMMTAN
from nb_hmm import NaiveBayesHMM

print("Start Testing")

def convert_observations_to_nbhmm(o_history):
    observations = []
    for key in sorted(o_history[0].keys()):
        obs_slice = []
        for i in range(len(o_history)):
            obs_slice.append(o_history[i][key])
        observations.append(obs_slice)
    return observations


def test(config):
    num_tests = config['num_tests']
    epoches = config['epoches']
    num_tests = config['num_tests']
    train_instance_size = config['train_instance_size']
    num_test_instances = config['num_test_instances']
    test_instance_size = config['test_instance_size']
    epoches = config['epoches']

    # test case
    class_name = "A"
    class_states_count = 4
    root_node = "B"
    nodes_info = {"B": 2, "C": 2, "D": 3, "E": 4, "F": 4}

    test_results = []
    for test_number in range(num_tests):
        # simulation models
        model_sim_1 = HMMTAN.initialize(
            class_name, class_states_count, nodes_info, root_node)
        model_sim_2 = HMMTAN.initialize(
            class_name, class_states_count, nodes_info, root_node)

        # generate samples from simulation model
        s_history, o_history_1 = model_sim_1.simulate(train_instance_size)
        train_df_1 = pd.DataFrame.from_dict(o_history_1)

        s_history, o_history_2 = model_sim_2.simulate(train_instance_size)
        train_df_2 = pd.DataFrame.from_dict(o_history_2)

        # train two new models with these data
        model_train_1 = HMMTAN.initialize(
            class_name, class_states_count, nodes_info, root_node)
        model_train_2 = HMMTAN.initialize(
            class_name, class_states_count, nodes_info, root_node)

        s1 = model_train_1.train(train_df_1, epoches=epoches, verbose=1)
        s2 = model_train_2.train(train_df_2, epoches=epoches, verbose=1)

        observations_nb_1 = convert_observations_to_nbhmm(o_history_1)
        observations_nb_2 = convert_observations_to_nbhmm(o_history_2)

        states = list(range(class_states_count))
        observables = [list(range(item[1])) for item in nodes_info.items()]

        model_nb_1 = NaiveBayesHMM.initialize(states, observables)
        model_nb_1.train(observations_nb_1, epoches)

        model_nb_2 = NaiveBayesHMM.initialize(states, observables)
        model_nb_2.train(observations_nb_2, epoches)

        results = []
        for i in range(num_test_instances):
            s_history, test_1 = model_sim_1.simulate(test_instance_size)
            s_history, test_2 = model_sim_2.simulate(test_instance_size)
            test_df_1 = pd.DataFrame.from_dict(test_1)
            test_df_2 = pd.DataFrame.from_dict(test_2)
            test_nb_1 = convert_observations_to_nbhmm(test_1)
            test_nb_2 = convert_observations_to_nbhmm(test_2)
            results.append({
                "label": 1,
                "hmmtan_1": model_train_1.score(test_df_1),
                "hmmtan_2": model_train_2.score(test_df_1),
                "hmmnb_1": model_nb_1.score(test_nb_1),
                "hmmnb_2": model_nb_2.score(test_nb_1),

            })
            results.append({
                "label": 2,
                "hmmtan_1": model_train_1.score(test_df_2),
                "hmmtan_2": model_train_2.score(test_df_2),
                "hmmnb_1": model_nb_1.score(test_nb_2),
                "hmmnb_2": model_nb_2.score(test_nb_2),

            })

        results_df = pd.DataFrame.from_dict(results)

        results_df['pred_tan'] = 2
        results_df.loc[results_df['hmmtan_1']
                               > results_df['hmmtan_2'], 'pred_tan'] = 1

        results_df['pred_nb'] = 2
        results_df.loc[results_df['hmmnb_1']
                              > results_df['hmmnb_2'], 'pred_nb'] = 1

        tan_accuracy = (results_df['pred_tan'] ==
                        results_df['label']).sum()/results_df.shape[0]
        nb_accuracy = (results_df['pred_nb'] ==
                       results_df['label']).sum()/results_df.shape[0]

        print("Test {} - hmmtan accuracy ={}% | hmmnb accuracy={}%".format(test_number,
              tan_accuracy*100, nb_accuracy*100))
        test_results.append(
            {"test_id": test_number, "tan_accuracy": tan_accuracy, "nb_accuracy": nb_accuracy})

    return test_results


configs = [
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 100,
        "num_test_instances": 20,
        "test_instance_size": 5,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 100,
        "num_test_instances": 20,
        "test_instance_size": 10,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 100,
        "num_test_instances": 20,
        "test_instance_size": 20,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 500,
        "num_test_instances": 20,
        "test_instance_size": 5,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 500,
        "num_test_instances": 20,
        "test_instance_size": 10,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 500,
        "num_test_instances": 20,
        "test_instance_size": 20,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 1000,
        "num_test_instances": 20,
        "test_instance_size": 5,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 1000,
        "num_test_instances": 20,
        "test_instance_size": 10,
    },
    {
        "num_tests": 100,
        "epoches": 20,
        "train_instance_size": 1000,
        "num_test_instances": 20,
        "test_instance_size": 20,
    }

]

test_results = []
for test_index, config in enumerate(configs):
    print("Test {} - with config {}".format(test_index, config))
    test_result = test(config)
    test_results.append({"test_id": test_index, "config": config, "test_result": test_result})


import pickle
with open("single_training_test.pickle", "wb") as f:
    pickle.dump(test_results, f)
