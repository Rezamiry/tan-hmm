import random
import pandas as pd
import numpy as np

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from hmmtan import HMMTAN
from nb_hmm import NaiveBayesHMM

# test case 
class_name = "A"
class_states_count = 2
root_node = "B"
nodes_info = {"B": 2, "C": 4, "D": 2, "E": 3, "F": 4}


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
    num_training_instances = config['num_training_instances']
    train_instance_size = config['train_instance_size']
    num_test_instances = config['num_test_instances']
    test_instance_size = config['test_instance_size']
    epoches = config['epoches']
    

    test_results = []
    for test_number in range(num_tests):
        # generate samples from simulation model
        tan_train_list_1 = []
        tan_train_list_2 = []
        nb_train_list_1 = []
        nb_train_list_2 = []
        # simulation model 1
        model_sim_1 = HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node)

        # simulation model 2
        model_sim_2 = HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node)
        
        for i in range(num_training_instances):
            s_history, o_history_1 = model_sim_1.simulate(train_instance_size)
            train_df_1 = pd.DataFrame.from_dict(o_history_1)
            observations_nb_1 = convert_observations_to_nbhmm(o_history_1)
            tan_train_list_1.append(train_df_1)
            nb_train_list_1.append(observations_nb_1)

            s_history, o_history_2 = model_sim_2.simulate(train_instance_size)
            train_df_2 = pd.DataFrame.from_dict(o_history_2)
            observations_nb_2 = convert_observations_to_nbhmm(o_history_2)
            tan_train_list_2.append(train_df_2)
            nb_train_list_2.append(observations_nb_2)

        # train two new models with these data
        model_train_1 = HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node)
        model_train_2 = HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node)

        s1 = model_train_1.train_multiple_observations(tan_train_list_1, epoches=epoches, verbose=0)
        s2 = model_train_2.train_multiple_observations(tan_train_list_2, epoches=epoches, verbose=0)


        states = list(range(class_states_count))
        observables = [list(range(item[1])) for item in nodes_info.items()]

        model_nb_1 = NaiveBayesHMM.initialize(states, observables)
        model_nb_1.train_multiple_observations(nb_train_list_1, epoches)

        model_nb_2 = NaiveBayesHMM.initialize(states, observables)
        model_nb_2.train_multiple_observations(nb_train_list_2, epoches)

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
        results_df['pred_tan'][results_df['hmmtan_1'] > results_df['hmmtan_2']] = 1

        results_df['pred_nb'] = 2
        results_df['pred_nb'][results_df['hmmnb_1'] > results_df['hmmnb_2']] = 1

        tan_accuracy = (results_df['pred_tan'] == results_df['label']).sum()/results_df.shape[0]
        nb_accuracy = (results_df['pred_nb'] == results_df['label']).sum()/results_df.shape[0]

        print("Test {} - hmmtan accuracy ={}% | hmmnb accuracy={}%".format(test_number, tan_accuracy*100, nb_accuracy*100))
        test_results.append({"test_id": test_number, "tan_accuracy": tan_accuracy, "nb_accuracy": nb_accuracy})

    return test_results


configs =  [{
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 10,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 5,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 10,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 10,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 10,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 20,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 5,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 10,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 20,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 10,
        "num_test_instances": 100,
        "test_instance_size": 10,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 5,
        "num_test_instances": 100,
        "test_instance_size": 5,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 5,
        "num_test_instances": 100,
        "test_instance_size": 10,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 20,
        "num_training_instances": 5,
        "num_test_instances": 100,
        "test_instance_size": 20,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 100,
        "num_training_instances": 2,
        "num_test_instances": 100,
        "test_instance_size": 5,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 100,
        "num_training_instances": 2,
        "num_test_instances": 100,
        "test_instance_size": 10,
    },
            {
        "num_tests": 100,
        "epoches": 10,
        "train_instance_size": 100,
        "num_training_instances": 2,
        "num_test_instances": 100,
        "test_instance_size": 20,
    }
            
]

test_results = []
for test_index, config in enumerate(configs):
    print("Test {} - with config {}".format(test_index, config))
    test_result = test(config)
    test_results.append({"test_id": test_index, "config": config, "test_result": test_result})


import pickle
with open("multiple_training_test.pickle", "wb") as f:
    pickle.dump(test_results, f)