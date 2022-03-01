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
dataFrame['Cloud9amCat'] = dataFrame['Cloud9am'].apply(
    lambda x: '1.>4' if x > 4 else '0.<=4')
dataFrame['Cloud3pmCat'] = dataFrame['Cloud3pm'].apply(
    lambda x: '1.>4' if x > 4 else '0.<=4')
dataFrame['Humidity9amCat'] = dataFrame['Humidity9am'].apply(
    lambda x: '1.>60' if x > 60 else '0.<=60')
dataFrame['Humidity3pmCat'] = dataFrame['Humidity3pm'].apply(
    lambda x: '1.>60' if x > 60 else '0.<=60')
dataFrame['RainfallCat'] = dataFrame['Rainfall'].apply(
    lambda x: '1.>15' if x > 15 else '0.<=15')
dataFrame['SunshineCat'] = dataFrame['Sunshine'].apply(
    lambda x: '1.>6' if x > 6 else '0.<=6')
dataFrame['Temp9amCat'] = dataFrame['Temp9am'].apply(
    lambda x: '1.>20' if x > 20 else '0.<=20')
dataFrame['Temp3pmCat'] = dataFrame['Temp3pm'].apply(
    lambda x: '1.>20' if x > 20 else '0.<=20')
dataFrame['WindGustSpeedCat'] = dataFrame['WindGustSpeed'].apply(
    lambda x: '0.<=40' if x <= 40 else '1.>40')


def probs(data, child, parent1=None, parent2=None, parent3=None):
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
            if parent3 == None:
                # Caclucate probabilities
                prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                                   normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                # Caclucate probabilities
                prob = pd.crosstab([data[parent1], data[parent2], data[parent3]], data[child], margins=False,
                                   normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob


# Create nodes by using our earlier function to automatically calculate probabilities
C9am = BbnNode(Variable(0, 'C9am', [' <= 4', ' > 4']), probs(
    dataFrame, child='Cloud9amCat', parent1='Temp9amCat', parent2='WindGustSpeedCat'))
C3pm = BbnNode(Variable(1, 'C3pm', [' <= 4', ' > 4']), probs(
    dataFrame, child='Cloud3pmCat', parent1='Temp3pmCat', parent2='Cloud9amCat', parent3='WindGustSpeedCat'))
H9am = BbnNode(Variable(2, 'H9am', [' <= 60', ' > 60']), probs(
    dataFrame, child='Humidity9amCat', parent1='RainfallCat'))
H3pm = BbnNode(Variable(3, 'H3pm', [' <= 60', ' > 60']), probs(
    dataFrame, child='Humidity3pmCat', parent1='Humidity9amCat'))
RainToday = BbnNode(Variable(4, 'RainToday', [' yes', ' no']), probs(
    dataFrame, child='RainToday'))
Rainfall = BbnNode(Variable(5, 'Rainfall', [' <= 15', ' > 15']), probs(
    dataFrame, child='RainfallCat', parent1='RainToday'))
Sunshine = BbnNode(Variable(6, 'Sunshine', [' <= 6', ' > 6']), probs(
    dataFrame, child='SunshineCat'))
Temp9am = BbnNode(Variable(7, 'Temp9am', [' <= 20', ' > 20']), probs(
    dataFrame, child='Temp9amCat', parent1='SunshineCat'))
Temp3pm = BbnNode(Variable(8, 'Temp3pm', [' <= 20', ' > 20']), probs(
    dataFrame, child='Temp3pmCat', parent1='Temp9amCat'))
WindGustSpeed = BbnNode(Variable(9, 'WindGustSpeed', [' <= 40', ' > 40']),
                        probs(dataFrame, child='WindGustSpeedCat'))
RainTomorrow = BbnNode(Variable(10, 'RainTomorrow', [' no ', ' yes']), probs(
    dataFrame, child='RainTomorrow', parent1='Humidity3pmCat', parent2='Cloud3pmCat'))

# Create Network
bbn = Bbn()\
    .add_node(C9am) \
    .add_node(C3pm) \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(RainToday) \
    .add_node(Rainfall) \
    .add_node(Sunshine) \
    .add_node(Temp9am) \
    .add_node(Temp3pm) \
    .add_node(RainTomorrow) \
    .add_node(WindGustSpeed) \
    .add_edge(Edge(RainToday, Rainfall, EdgeType.DIRECTED)) \
    .add_edge(Edge(Rainfall, H9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RainTomorrow, EdgeType.DIRECTED)) \
    .add_edge(Edge(Sunshine, Temp9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(Temp9am, Temp3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(Temp9am, C9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(C9am, C3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(C3pm, RainTomorrow, EdgeType.DIRECTED)) \
    .add_edge(Edge(WindGustSpeed, C9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(WindGustSpeed, C3pm, EdgeType.DIRECTED)) \

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# Define a function for printing marginal probabilities


def print_probs():
    for node in join_tree.get_bbn_nodes():
        if(node.to_dict()['variable']['id'] == 10):
            potential = join_tree.get_bbn_potential(node)
            print("Probabilities for RainTomorrow:")
            print(potential)


# Use the above function to print marginal probabilities
print_probs()
