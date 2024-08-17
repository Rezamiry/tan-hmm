import pickle
import pandas as pd
import numpy as np

from hmmtan import HMMTAN


print("Loading dataset")
levels = 6
pca = 5
print("pca: {} - levels: {}".format(pca, levels))
with open("train_test_pca_{}_level_{}.pickle".format(pca, levels), "rb") as f:
    dataset = pickle.load(f)


train_set = dataset["train_set"]
test_set = dataset["test_set"] 


class_name = "state"
class_states_count = 2
nodes_info = {}
for column in train_set['1'][0].columns:
    nodes_info[str(column)] = levels

root_node = train_set['1'][0].columns[0]

print("Initializing training")

models_dic = {}
for key, value in train_set.items():
    print("training {} model".format(key))
    model_train = HMMTAN.initialize(class_name, class_states_count, nodes_info, root_node)
    df_list = []
    for df in train_set[key]:
        df.columns = [str(col) for col in df.columns]
        df_list.append(df)
    model_train.train_multiple_observations(df_list, epoches=20, verbose=1)
    models_dic[key] = model_train

with open(f"tan_hmm_motion_trained_models_pca_{pca}_level_{levels}.pickle", "wb") as f:
    pickle.dump(models_dic, f)
