import sys
import pandas as p
import random as r
from collections import defaultdict
import numpy as np

class FeatureVector(object):
    def __init__(self, features, label):
        self.features = features
        self.label = label
    

def get_feature_vectors(data_frame):
    
    question_stub = "question"
    feature_names = ['diag']
    f_vectors = np.array([])
    
    for x in range(1,31):
        feature_names.append(question_stub + str(x))

    for x in range(data_frame.shape[0]):
        review = data_frame.iloc[x]
        features = np.zeros(len(feature_names)-1)
        i = 0
        for feature in feature_names[1:]:
            value = review[feature]
            features[i] = value
            i+=1

        
        label = int(review[feature_names[0]] == 'asd')
        f_vectors = np.append(f_vectors, FeatureVector(features=features, label=label))

    return f_vectors
    
def remove_errored_data(data_frame):
    to_keep = []
    question_stub = "question"
    
    new = p.DataFrame()
    
    for i in range(data_frame.shape[0]):
        error = False
        row = data_frame.iloc[i]
        for j in range(1,31):
            question = question_stub + str(j)
            value = row[question]
            
            if(value != value): #nan
                error = True
                break

        if(not error):
            new = new.append(row)

    
    return new
                
                
    
        
def downsample_dataset(data_frame):
    asd = data_frame[data_frame['diag'] == 'asd']
    non = data_frame[data_frame['diag'] == 'non-asd']

    random = set()
    while(len(random) < non.shape[0]):
        random.add(r.randint(0, asd.shape[0]-1))

    new = p.DataFrame()
    for rand in random:
        new = new.append(asd.iloc[rand])


    return p.concat([new, non], sort=False)

def upsample_dataset(data_frame):
    asd = data_frame[data_frame['diag'] == 'asd']
    non = data_frame[data_frame['diag'] == 'non-asd']
    
    new_non = p.DataFrame(non)
    
    # sample with replacement from the non-asd until the sizes are the same
    while(new_non.shape[0] < asd.shape[0]):

        idx = r.randint(0, non.shape[0]-1)
        
        new_row = non.iloc[idx]
        
        new_non = new_non.append(new_row, ignore_index=True)

    return p.concat([asd, new_non])

    

def get_scorer_to_child(dataset):
    scorer_to_child = defaultdict(set)

    for child, scorer in zip(dataset['child_id'], dataset['scorer_id']):
        scorer_to_child[scorer].add(child)

    return scorer_to_child
    
def get_valid_children(scorer_to_child):

    valid_children = set()
    to_remove = []
    for scorer in scorer_to_child:
        if(len(scorer_to_child[scorer]) < 20):
            to_remove.append(scorer)
    
    for removal in to_remove:
        del scorer_to_child[removal]
            
    scorers = list(scorer_to_child.keys())
    for x in range(len(scorers)-1):
        valid_children = scorer_to_child[scorers[x]].intersection(scorer_to_child[scorers[x+1]])

    return valid_children, scorers


def main(argv):
    # read in the data
    primary_dataset = p.read_csv("./Tariq-Wall-2018-PLOS-MEDICINE/datasets/primary_dataset.csv")
    validation_dataset = p.read_csv("./Tariq-Wall-2018-PLOS-MEDICINE/datasets/validation_dataset.csv")
    #
    
    # remove records with nan
    primary_dataset = remove_errored_data(primary_dataset)
    
    scorer_to_child_primary = get_scorer_to_child(primary_dataset)
    scorer_to_child_validation = get_scorer_to_child(validation_dataset)

    valid_children_primary, scorers_primary = get_valid_children(scorer_to_child_primary)
    valid_children_validation, scorers_validation = get_valid_children(scorer_to_child_validation)

    assert(len(scorers_primary) == len(scorers_validation))

    # upsample or downsample to deal with unequal class membership..
    #primary_dataset = downsample_dataset(primary_dataset)
    primary_dataset = upsample_dataset(primary_dataset)
    
    
    # equal class representation
    assert(primary_dataset[primary_dataset['diag'] == 'asd'].shape[0] == primary_dataset[primary_dataset['diag'] == 'non-asd'].shape[0])
    

    features_primary = get_feature_vectors(primary_dataset)

    print(features_primary[0].features, features_primary[0].label)

    
    
    
    

    

    
            
    
if(__name__ == '__main__'):
    main(sys.argv[1:])

