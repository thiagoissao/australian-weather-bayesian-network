import pandas as pd  # for data manipulation
import networkx as nx  # for drawing graphs
import matplotlib.pyplot as plt  # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Set Pandas options to display more columns
pd.options.display.max_columns = 50

# Read in the weather data csv
dataFrame = pd.read_csv('./dataset/weatherAUS-small.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
dataFrame = dataFrame[pd.isnull(dataFrame['RainTomorrow']) == False]

# For other columns with missing values, fill them in with column mean
dataFrame = dataFrame.fillna(dataFrame.mean())

# Create bands for variables that we want to use in the model
dataFrame['WindGustSpeedCat'] = dataFrame['WindGustSpeed'].apply(lambda x: '0.<=40' if x <= 40 else
                                                                 '1.40-50' if 40 < x <= 50 else '2.>50')
dataFrame['Humidity9amCat'] = dataFrame['Humidity9am'].apply(
    lambda x: '1.>60' if x > 60 else '0.<=60')
dataFrame['Humidity3pmCat'] = dataFrame['Humidity3pm'].apply(
    lambda x: '1.>60' if x > 60 else '0.<=60')


def probs(data, child, parent1=None, parent2=None):
    if parent1 == None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index(
        ).to_numpy().reshape(-1).tolist()
    elif parent1 != None:
        # Check if child node has 1 parent or 2 parents
        if parent2 == None:
            # Caclucate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Caclucate probabilities
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob


# Create nodes by using our earlier function to automatically calculate probabilities
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(
    dataFrame, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(
    dataFrame, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']),
            probs(dataFrame, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(
    dataFrame, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

# Create Network
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# Define a function for printing marginal probabilities


def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')


# Use the above function to print marginal probabilities
print_probs()
