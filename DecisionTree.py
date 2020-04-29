import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
from graphviz import Digraph
from collections import defaultdict
from prettytable import PrettyTable
sys.setrecursionlimit(10000)


def remove_nan(data_frame):
    question_stub = "question"
    to_remove = []
    for index in data_frame.index.tolist():
        valid = True
        row = data_frame.loc[index]
        for j in range(1,31):
            question = question_stub + str(j)
            value = row[question]
            if(value != value):
                valid = False
        if(not valid):
            to_remove.append(index)
    
    return to_remove

def remove_underrepresented_data(data_frame):
    child_to_counts = defaultdict(int)
    
    for x in range(data_frame.shape[0]):
        row = data_frame.iloc[x]
        child_to_counts[row['child_id']] += 1
    
    to_remove = []
    too_many = []
    for child, count in child_to_counts.items():
        if(count < 3):
            to_remove.append(child)
        
        if(count > 3):
            too_many.append(child)
            
    
    
    to_remove_index = []
    for index in data_frame.index.tolist():
        child = data_frame.loc[index]['child_id']
        if(child in to_remove):
            to_remove_index.append(index)
        
        if(child in too_many):
            to_remove_index.append(index)
            too_many.remove(child)
        
        
    
    return to_remove_index

def upsample(data_frame):
    min_pop = data_frame.loc[data_frame['diag'] == 'non-asd']
    maj_pop = data_frame.loc[data_frame['diag'] == 'asd']
    sample_size = maj_pop.shape[0] - min_pop.shape[0]
    
    indices = min_pop.index.tolist()
    
    # sample minority population with replacement
    sample = random.choices(population=indices, k=sample_size)
    
    return sample


def get_train_test():
    
    attributes_train = ['child_id', 'scorer_id', 'question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7', 'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14', 'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21', 'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28', 'question29', 'question30', 'diag']
    attributes_test = ['child_id', 'scorer_id', 'question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7', 'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14', 'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21', 'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28', 'question29', 'question30', 'ASD']
    train, test = pd.read_csv('./Tariq-Wall-2018-PLOS-MEDICINE/datasets/primary_dataset.csv'), pd.read_csv('./Tariq-Wall-2018-PLOS-MEDICINE/datasets/validation_dataset.csv')
    train, test = train[attributes_train], test[attributes_test]

    to_remove = remove_nan(train)
    train.drop(to_remove, inplace=True)

    to_remove = remove_underrepresented_data(train)
    train.drop(to_remove, inplace=True)

    to_remove = remove_underrepresented_data(test)
    test.drop(to_remove, inplace=True)
    child_to_index_train = get_children_to_index(train)
    child_to_index_test = get_children_to_index(test)
    #child_to_index_test = get_children_to_index(test)

    attributes_test = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7', 'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14', 'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21', 'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28', 'question29', 'question30', 'ASD']
    test = test[attributes_test]

    attributes_train = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7', 'question8', 'question9', 'question10', 'question11', 'question12', 'question13', 'question14', 'question15', 'question16', 'question17', 'question18', 'question19', 'question20', 'question21', 'question22', 'question23', 'question24', 'question25', 'question26', 'question27', 'question28', 'question29', 'question30', 'diag']
    train = train[attributes_train]

    
    sample = upsample(train)

    for idx in sample: # hacky way to append to pandas dataframe without copying..
        train.loc[train.index.max() + 1] = train.loc[idx]


    data_train, data_test = train.values, test.values

    return train, test, data_train, data_test, child_to_index_train, child_to_index_test


def get_children_to_index(data_frame):
    child_to_index = defaultdict(list)
    
    for index in range(data_frame.shape[0]):
        row = data_frame.iloc[index]
        child_to_index[row['child_id']].append(index)
    
    
    return child_to_index


#feature selection for 4, and 8 features
LABEL_IDX = -1
def is_single_class(data):
    return len(np.unique(data[:,LABEL_IDX])) == 1


def classify(data):
    classes, counts = np.unique(data[:, LABEL_IDX], return_counts=True)
    return int(classes[np.argmax(counts)] == 'asd')

def get_data_partitions(data, features=None):
    
    #all features
    if(features == None):
        features = np.arange(0, data.shape[1]-1)
        
    partitions = defaultdict(list)
    for idx in range(0, data.shape[1]-1):
        # one of selected features?
        #print(features)
        if(idx in features):
            
            unique_scores = np.unique(data[:, idx])
            '''
            if(len(unique_scores) == 1): # edge case what if unique score == 1
                partitions[idx].append(unique_scores[0])
            '''
            for x in range(len(unique_scores)-1):
                divide = np.mean(unique_scores[x:x+2])
                partitions[idx].append(divide)
        else:
            continue
        
    return partitions

def partition_data(data, question, score):
    a,b = data[data[:, question] >= score], data[data[:, question] < score]
    
    return a,b

def get_entropy(data):
    _, counts = np.unique(data[:, LABEL_IDX], return_counts=True)
    p_i = counts / sum(counts)
    return (p_i * -np.log2(p_i)).sum() # entropy sum(p_i * -log2(p_i))


def get_total_entropy(a, b):
    n = a.shape[0] + b.shape[0]
    p_a, p_b = a.shape[0] / n, b.shape[0] / n
    entropy_a, entropy_b = get_entropy(a), get_entropy(b)
    return p_a*entropy_a + p_b*entropy_b


def lowest_entropy_partition(data, data_partitions):
    
    lowest = np.inf
    split_question = None
    split_score = None
    
    for question in data_partitions:
        cutoffs = data_partitions[question]
        for score in cutoffs:
            a,b = partition_data(data, question, score)
            entropy = get_total_entropy(a,b)
            if(entropy < lowest):
                lowest = entropy
                split_question = question
                split_score = score
    
    return split_question, split_score, lowest

def select_k_lowest_features(data, k, cost_function='Entropy'):
    features = set()
    q = Queue()
    q.enqueue(data)
    while(len(features) < k):
        size = q.size()
        for x in range(size):
            data = q.dequeue()
            
            data_partitions = get_data_partitions(data)
            
            if(len(features) == k):
                break
            feature, value = None, None
            
            if(cost_function == 'Entropy'):
                feature, value, _ = lowest_entropy_partition(data, data_partitions)
            elif(cost_function == 'Gini'):
                feature, value, _ = lowest_gini_split(data, data_partitions)
            
            if(feature != None):
                features.add(feature)
                a, b = partition_data(data, feature, value)
                q.enqueue(a)
                q.enqueue(b)
            else:
                #print(data, data_partitions)
                continue
    features = list(features)
    #print(len(features))
    return features[0:k+1]

class Queue(object):
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        self.queue[0], self.queue[-1] = self.queue[-1], self.queue[0]
        
        return self.queue.pop()
    
    def peek(self):
        if(not self.is_empty()):
            return self.queue[0]
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def size(self):
        return len(self.queue)

class TreeNode(object):
    def __init__(self, feature, value, id=None):
        self.question = "%s >= %0.2f" % (feature, value)
        self.feature = feature
        self.value = value
        self.yes = None
        self.no = None
        self.id = id
    
    def __str__(self):
        return self.question
    
    def set_id(self, id):
        self.id = id
    def get_id(self):
        return str(self.id)
    

def get_decision_tree(data, features=None):
    
    if(is_single_class(data)):
        return classify(data)
    else:
        # get question, and cutoff with lowest overall entropy
        data_partitions = get_data_partitions(data, features)
        split_question, split_value, lowest_entropy = lowest_entropy_partition(data, data_partitions)
        
        # no way to split the data with given features.. so return best guess
        if(split_question == None):
            #print("Model performing best guess for leaf node..")
            return classify(data)
            '''
            for x in range(0, data.shape[1]-1):
                if(x in features):
                    print(data[:,x].shape)
                    unique_scores = np.unique(data[:, x])
                    print(x, data[:,x])
                    print(unique_scores)
            '''     
        node = TreeNode(split_question, split_value)
        a, b = partition_data(data, split_question, split_value)
        
        # recurse on left, and right subtrees..
        node.yes = get_decision_tree(a, features)
        node.no = get_decision_tree(b, features)
        
        return node
    
    
    
def bfs(root, title="Default Title"):
    q = Queue()
    G = Digraph(comment=title)
    
    cur_id = 0
    q.enqueue(root)
    G.node(str(cur_id), root.question)
    root.set_id(cur_id)
    cur_id += 1
    while(not q.is_empty()):
        size = q.size()
        
        # process that lvl
        for x in range(size):
            node = q.dequeue()
            
            #print(node, end='\t\t')
            # enqueue children if not classification
            if(type(node) != int):
                # build up the graph..
                if(type(node.yes) != int):
                    node.yes.set_id(cur_id)
                    G.node(str(cur_id), node.yes.question)
                    G.edge(node.get_id(), node.yes.get_id(), label='y')
                else:
                    G.node(str(cur_id), str(node.yes), shape='square')
                    G.edge(node.get_id(), str(cur_id), label='y')
                cur_id += 1
                
                
                if(type(node.no) != int):
                    node.no.set_id(cur_id)
                    G.node(str(cur_id), str(node.no))
                    G.edge(node.get_id(), node.no.get_id(), label='n')
                else:
                    G.node(str(cur_id), str(node.no), shape='square')
                    G.edge(node.get_id(), str(cur_id), label='n')
                    
                cur_id += 1
                
                    
                q.enqueue(node.yes)
                q.enqueue(node.no)
            else:
                continue
            
            
        # next level please..
        #print("\n\n")
    
    return G

def classify_obs(root, obs):
    cur = root
    while(type(cur) != int):
        #print(cur)
        feature, value = cur.feature, cur.value
        direction = obs[feature] >= value
        if(direction == True):
            #print("Yes")
            cur = cur.yes
        else:
            #print("No")
            cur = cur.no
    
    return cur

def get_predictions(root, data_test):
    preds = []
    for obs in data_test:
        pred = classify_obs(root, obs)
        preds.append(pred)
    return np.array(preds)

def vote(preds):
    unique, counts = np.unique(preds, return_counts=True)
    return unique[np.argmax(counts)]



def child_to_prediction_validate(child_to_index, data_pred):
    
    child_to_pred = defaultdict(int)
    
    for child in child_to_index:
        preds = []
        indices = child_to_index[child]
        for index in indices:
            
            preds.append(data_pred[index])
        pred = vote(np.array(preds))
        child_to_pred[child] = pred
    
    return child_to_pred


def get_child_to_prediction(child_to_index, data):
    
    child_to_pred = defaultdict(int)
    
    for child in child_to_index:
        preds = []
        indices = child_to_index[child]
        for index in indices:
            actual = int(data.iloc[index]['ASD'] == 1)
            preds.append(actual)
            
            
        child_to_pred[child] = vote(preds)
    
    return child_to_pred

def validate_votes(actual, pred):
    correct = 0
    for c1, c2 in zip(pred, actual):
        if(c1 == c2):
            p1, p2 = pred[c1], actual[c2]
            if(p1 == p2):
                correct += 1
    
    return correct / len(pred)

def confusion_matrix(data_pred, data):
    mapping = {'asd': 1, 'non-asd': 0, 0:0, 1:1}
    labels = data[:, LABEL_IDX]
    matrix = np.zeros(4).reshape(2,2)
    #print(labels)
    #print(data_pred)
    for pred, obs in zip(data_pred, labels):
        #print(pred, obs)
        matrix[int(pred)][int(mapping[obs])] += 1
    
    return matrix


def get_sensitivity(confusion_matrix):
    return confusion_matrix[1, 1] / sum(confusion_matrix[1,:])

def get_specificity(confusion_matrix):
    bottom = sum(confusion_matrix[0, :])
    
    return confusion_matrix[0, 0] / bottom


def gini_index(data):
    LABEL_IDX = -1
    mappings = {'asd': 1, 'non-asd':0, 0:0, 1:1}
    unique, counts = np.unique(data[:, LABEL_IDX], return_counts=True)
    
    total = sum(counts)
    gini_score = 1 - np.sum((counts / total)**2) # 1 - sum(p_i)^2
    
    return gini_score

def weighted_gini_score(a, b):
    n = a.shape[0] + b.shape[0]
    
    p_a, p_b = a.shape[0] / n, b.shape[0] / n
    
    gini_a, gini_b = gini_index(a), gini_index(b)
    
    return p_a*gini_a + p_b*gini_b

def sorted_gini_splits(data, partitions, k):
    
    split_to_gini = defaultdict(float)
    
    for question, values in partitions.items():
        for score in values:
            a,b = partition_data(data, question, score)
            gini_score = weighted_gini_score(a,b)
            
            split_to_gini[(question, score)] = gini_score
    
    return sorted(split_to_gini.items(), key=lambda x:x[1])[0:k]
    
def lowest_gini_split(data, partitions):
    
    lowest = np.inf
    winning_score = None
    winning_question = None
    
    for question, values in partitions.items():
        for score in values:
            a,b = partition_data(data, question, score)
            
            gini_score = weighted_gini_score(a,b)
            
            if(gini_score < lowest):
                lowest = gini_score
                winning_score = score
                winning_question = question
                
            
    return winning_question, winning_score, lowest


def cart_algorithm(data, features=None):
    
    if(is_single_class(data)):
        return classify(data)
    
    #print(features)
    partitions = get_data_partitions(data, features)
    question, value, gini = lowest_gini_split(data, partitions)
    
    # no way to further split the data
    if(question is None):
        return classify(data)
    
    a,b = partition_data(data, question, value)
    
    node = TreeNode(feature=question, value=value)
    
    node.yes = cart_algorithm(a, features)
    node.no = cart_algorithm(b, features)
    
    return node

def run_trials(num_trials, min_feature, max_feature, cost_function='Entropy', all_features=False):
    
    feature_to_sensitivity = defaultdict(list)
    feature_to_specificity = defaultdict(list)
    feature_to_accuracy = defaultdict(list)
    feature_to_uar = defaultdict(list)
    
    trees = defaultdict(list)
    trees_g = defaultdict(list)
    for trial in range(num_trials):
        '''
        if(trial % 100 == 0):
        
        print("Trial %d" % (trial + 1))
        '''
        # read data, perform cleaning, upsampling, etc..
        train, test, data_train, data_test, child_to_index_train, child_to_index_test = get_train_test()
        
        for num_features in range(min_feature, max_feature + 1):
            
            features = None
            # get k most important features
            if(not all_features):
                features = select_k_lowest_features(data_train, num_features, cost_function)
            #print(num_features)
            #print(features)
            # build decision tree, get root back in tree
            tree = None
            if(cost_function == 'Entropy'):
                tree = get_decision_tree(data_train, features=features)
            elif(cost_function == 'Gini'):
                tree = cart_algorithm(data_train, features=features)
            
            trees[num_features].append(tree)
            # get graphical representation of decision tree
            tree_g = bfs(tree)
            trees_g[num_features].append(tree_g)

            # validate model on test data
            pred_test = get_predictions(tree, data_test)

            pred_test = child_to_prediction_validate(child_to_index_test, pred_test)

            actual_test = get_child_to_prediction(child_to_index_test, test)

            pred = np.array(list(pred_test.values())).reshape(len(actual_test), 1)

            actual = np.array(list(actual_test.values())).reshape(len(actual_test), 1)

            # get the confusion matrix
            c_matrix = confusion_matrix(pred, actual)

            # get sensitivity, specificity, accuracy, and unweighted average recall
            sensitivity = get_sensitivity(c_matrix)

            specificity = get_specificity(c_matrix)

            accuracy = validate_votes(actual_test, pred_test)
            
            uar = (sensitivity + specificity) / 2
            
            feature_to_sensitivity[num_features].append(sensitivity)
            feature_to_specificity[num_features].append(specificity)
            feature_to_accuracy[num_features].append(accuracy)
            feature_to_uar[num_features].append(uar)
    
    
    return feature_to_sensitivity, feature_to_specificity, feature_to_accuracy, feature_to_uar, trees, trees_g


def confidence_interval_95(values):
    z = 1.960
    x_mean = np.mean(values)
    x_std = np.std(values)
    n = len(values)
    
    confidence = z * (x_std / np.sqrt(n))
    
    return confidence

def generate_performance_plots(feature_to_sensitivity, feature_to_specificity, feature_to_accuracy, feature_to_uar, cost_function):
    x, y = [], []
    ubs, lbs = [], []
    for num_feature, values in sorted(feature_to_specificity.items(), key=lambda x:x[0]):
        x.append(num_feature)
        y.append(np.mean(values))
        confidence = confidence_interval_95(values)
        ubs.append(np.mean(values) + confidence)
        lbs.append(np.mean(values) - confidence)

    plt.plot(x, y, marker=',', linewidth=3, color='red')
    plt.fill_between(x, ubs, lbs, color='green', alpha=.10)
    plt.xlabel("# features")
    plt.ylabel("Percent (%)")
    plt.title("Specificity with {}".format(cost_function))
    plt.grid(color='lightgray')
    plt.show()

    x, y = [], []
    ubs, lbs = [], []
    for num_feature, values in sorted(feature_to_sensitivity.items(), key=lambda x:x[0]):
        x.append(num_feature)
        y.append(np.mean(values))
        confidence = confidence_interval_95(values)
        ubs.append(np.mean(values) + confidence)
        lbs.append(np.mean(values) - confidence)

    plt.plot(x, y, marker=',', linewidth=3, color='red')
    plt.fill_between(x, ubs, lbs, color='green', alpha=.10)
    plt.xlabel("# features")
    plt.ylabel("Percent (%)")
    plt.title("Sensitivity with {}".format(cost_function))
    plt.grid(color='lightgray')
    plt.show()

    x, y = [], []
    ubs, lbs = [], []
    for num_feature, values in sorted(feature_to_uar.items(), key=lambda x:x[0]):
        x.append(num_feature)
        y.append(np.mean(values))
        confidence = confidence_interval_95(values)
        ubs.append(np.mean(values) + confidence)
        lbs.append(np.mean(values) - confidence)

    plt.plot(x, y, marker=',', linewidth=3, color='red')
    plt.fill_between(x, ubs, lbs, color='green', alpha=.10)
    plt.xlabel("# features")
    plt.ylabel("Percent (%)")
    plt.title("Unweighted Average Recall with {}".format(cost_function))
    plt.grid(color='lightgray')
    plt.show()

    x, y = [], []
    ubs, lbs = [], []
    for num_feature, values in sorted(feature_to_accuracy.items(), key=lambda x:x[0]):
        x.append(num_feature)
        y.append(np.mean(values))
        confidence = confidence_interval_95(values)
        ubs.append(np.mean(values) + confidence)
        lbs.append(np.mean(values) - confidence)

    plt.plot(x, y, marker=',', linewidth=3, color='red')
    plt.fill_between(x, ubs, lbs, color='green', alpha=.10)
    plt.xlabel("# features")
    plt.ylabel("Percent (%)")
    plt.title("Accuracy with {}".format(cost_function))
    plt.grid(color='lightgray')
    plt.show()
    
    
def get_weighted_feature_to_counts(trees_dict):
    feature_to_counts = defaultdict(int)
    for feature in trees_dict:
        trees = trees_dict[feature]
        #print(feature)
        for tree in trees:
            # perform bfs and count feature occurance
            
            level = 0
            q = Queue()
            q.enqueue(tree)
            while(not q.is_empty()):
                size = q.size()
                for x in range(size):
                    cur_node = q.dequeue()
                    question = cur_node.feature

                    if(cur_node.no is not None):

                        if(type(cur_node.no) != int):
                            q.enqueue(cur_node.no)

                    if(cur_node.yes is not None):

                        if(type(cur_node.yes) != int):
                            q.enqueue(cur_node.yes)

                    feature_to_counts[question] += 1/(2**level) # weight count by 1/2^level

                # next level
                level += 1
                
        
    
    # normalize to probabilities
    total_counts = sum(feature_to_counts.values())
    feature_to_counts = {feature: count / total_counts for feature, count in feature_to_counts.items()}
    return feature_to_counts

def get_prob_vector(feature_counts):
    probs = []
    for x in range(30):
        if(x in feature_counts.keys()):
            probs.append(feature_counts[x])
        else:
            probs.append(0)
    
    return np.array(probs)

def sorted_feature_table(features):
    table = PrettyTable()
    table.field_names = ["Feature", "Probability"]
    for feature, probability in features:
        table.add_row([feature, probability])
    
    return table


def random_partition(data_partitions, probabilities=None):
    
    # get random feature
    features, random_feature = None, None
    if(probabilities is None):
        features = list(data_partitions.keys())
        random_feature = features[random.randint(0, len(features)-1)]
    else:
        features = np.arange(0,30)
        random_feature = np.random.choice(30, 1, p=probabilities)[0]
    
    # get random value for that feature
    values = data_partitions[random_feature]
    random_value = values[random.randint(0, len(values)-1)]
    
    return random_feature, random_value

def get_forest_classification(df, data, forest, child_to_index_test):
    
    actual_test = get_child_to_prediction(child_to_index_test, df)
    actual = np.array(list(actual_test.values())).reshape(len(actual_test), 1)
    
    consensus_predictions = defaultdict(int)
    preds = []
    for tree in forest:
        #print(tree)
        pred_test = get_predictions(tree, data)
        pred_test = child_to_prediction_validate(child_to_index_test, pred_test)
        
        pred = np.array(list(pred_test.values())).reshape(len(actual_test), 1)
        
        # get the confusion matrix
        #c_matrix = confusion_matrix(pred, actual)
        pred = np.squeeze(pred.reshape(1, len(pred)))
        preds.append(pred)
        
    
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            consensus_predictions[j] += preds[i][j]
    
    final_predictions = []
    for patient, votes in sorted(consensus_predictions.items(), key= lambda x: x[0]):
        final_predictions.append(int(votes >= (len(forest) // 2)))
    
    final_predictions = np.array(final_predictions).reshape(len(final_predictions), 1)
    
    return final_predictions, actual

def get_random_decision_tree(data, features=None, probabilities=None):
    
    if(is_single_class(data)):
        return classify(data)
    else:
        # get question, and cutoff with lowest overall entropy
        data_partitions = get_data_partitions(data, features)
        split_question, split_value = random_partition(data_partitions)
        
        # no way to split the data with given features.. so return best guess
        if(split_question == None):
            #print("Model performing best guess for leaf node..")
            return classify(data)
            '''
            for x in range(0, data.shape[1]-1):
                if(x in features):
                    print(data[:,x].shape)
                    unique_scores = np.unique(data[:, x])
                    print(x, data[:,x])
                    print(unique_scores)
            '''     
        node = TreeNode(split_question, split_value)
        a, b = partition_data(data, split_question, split_value)
        
        # recurse on left, and right subtrees..
        node.yes = get_decision_tree(a, features)
        node.no = get_decision_tree(b, features)
        
        return node


def random_cart_algorithm(data, features=None, probabilities=None):
    
    if(is_single_class(data)):
        return classify(data)
    
    #print(features)
    data_partitions = get_data_partitions(data, features)
    question, value = random_partition(data_partitions)
    
    # no way to further split the data
    if(question is None):
        return classify(data)
    
    a,b = partition_data(data, question, value)
    
    node = TreeNode(feature=question, value=value)
    
    node.yes = cart_algorithm(a, features)
    node.no = cart_algorithm(b, features)
    
    return node

def random_forest_trials(num_trees, num_trials, min_feature, max_feature, cost_function='Entropy', probabilities=None):
    
    feature_to_sensitivity = defaultdict(list)
    feature_to_specificity = defaultdict(list)
    feature_to_accuracy = defaultdict(list)
    feature_to_uar = defaultdict(list)
    
    for num_features in range(min_feature, max_feature + 1):

        for trial in range(num_trials):
            
            #print("Feature %d, Trial %d" % (num_features, trial+1))
            train, test, data_train, data_test, child_to_index_train, child_to_index_test = get_train_test()

            forest = []
            for x in range(num_trees):
                
                features = None
                if(probabilities is None):
                    features = select_k_lowest_features(data_train, num_features, cost_function)
                
                if(cost_function == 'Entropy'):
                    tree = get_random_decision_tree(data_train, features, probabilities)
                    forest.append(tree)
                elif(cost_function == 'Gini'):
                    tree = random_cart_algorithm(data_train, features, probabilities)
                    forest.append(tree)

            voted_preds, actual = get_forest_classification(test, data_test, forest, child_to_index_test)
            
            #print(voted_preds, actual)
            c_matrix = confusion_matrix(voted_preds, actual)
            #print(c_matrix)
            # get sensitivity, specificity, accuracy, and unweighted average recall
            sensitivity = get_sensitivity(c_matrix)
            
            specificity = get_specificity(c_matrix)
            #print(specificity)

            accuracy = get_accuracy(c_matrix)

            uar = (sensitivity + specificity) / 2
            
            feature_to_sensitivity[num_features].append(sensitivity)
            feature_to_specificity[num_features].append(specificity)
            feature_to_accuracy[num_features].append(accuracy)
            feature_to_uar[num_features].append(uar)
    
    
    return feature_to_sensitivity, feature_to_specificity, feature_to_accuracy, feature_to_uar


def get_accuracy(confusion_matrix):
    correct = sum(np.diagonal(confusion_matrix))
    total = sum(sum(confusion_matrix))
    
    return correct / total