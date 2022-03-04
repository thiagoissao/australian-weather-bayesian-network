import pandas as pd  # for data manipulation

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.graph.jointree import EvidenceBuilder
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


moreThan4 = '>4'
lessOrEqualThan4 = '<=4'

moreThan60 = '>60'
lessOrEqualThan60 = '<=60'

moreThan40 = '>40'
lessOrEqualThan40 = '<=40'

yes = 'yes'
no = 'no'

moreThan15 = '>15'
lessOrEqualThan15 = '<=15'

moreThan6 = '>6'
lessOrEqualThan6 = ' =6'

moreThan20 = '>20'
lessOrEqualThan20 = '<=20'

# Create nodes by using our earlier function to automatically calculate probabilities
C9am = BbnNode(Variable(0, 'C9am', [lessOrEqualThan4, moreThan4]), probs(
    dataFrame, child='Cloud9amCat', parent1='WindGustSpeedCat'))
C3pm = BbnNode(Variable(1, 'C3pm', [lessOrEqualThan4, moreThan4]), probs(
    dataFrame, child='Cloud3pmCat', parent1='WindGustSpeedCat', parent2='Cloud9amCat'))

H9am = BbnNode(Variable(2, 'H9am', [lessOrEqualThan60, moreThan60]), probs(
    dataFrame, child='Humidity9amCat', parent1='RainfallCat', parent2='Temp9amCat'))
H3pm = BbnNode(Variable(3, 'H3pm', [lessOrEqualThan60, moreThan60]), probs(
    dataFrame, child='Humidity3pmCat', parent1='Humidity9amCat', parent2='Temp3pmCat'))
RainToday = BbnNode(Variable(4, 'RainToday', [no, yes]), probs(
    dataFrame, child='RainToday'))
Rainfall = BbnNode(Variable(5, 'Rainfall', [lessOrEqualThan15, moreThan15]), probs(
    dataFrame, child='RainfallCat', parent1='RainToday'))
Sunshine = BbnNode(Variable(6, 'Sunshine', [lessOrEqualThan6, moreThan6]), probs(
    dataFrame, child='SunshineCat'))
Temp9am = BbnNode(Variable(7, 'Temp9am', [lessOrEqualThan20, moreThan20]), probs(
    dataFrame, child='Temp9amCat', parent1='SunshineCat'))
Temp3pm = BbnNode(Variable(8, 'Temp3pm', [lessOrEqualThan20, moreThan20]), probs(
    dataFrame, child='Temp3pmCat', parent1='Temp9amCat'))
WindGustSpeed = BbnNode(Variable(9, 'WindGustSpeed', [lessOrEqualThan40, moreThan40]),
                        probs(dataFrame, child='WindGustSpeedCat'))
RainTomorrow = BbnNode(Variable(10, 'RainTomorrow', [no, yes]), probs(
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
    .add_edge(Edge(Temp9am, H9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(Temp3pm, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(C9am, C3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(C3pm, RainTomorrow, EdgeType.DIRECTED)) \
    .add_edge(Edge(WindGustSpeed, C9am, EdgeType.DIRECTED)) \
    .add_edge(Edge(WindGustSpeed, C3pm, EdgeType.DIRECTED)) \

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# Define a function for printing marginal probabilities


def print_probs():
    for node in join_tree.get_bbn_nodes():
        # if(node.to_dict()['variable']['id'] == 10):
        potential = join_tree.get_bbn_potential(node)
        print("Probabilities for RainTomorrow:")
        print(potential)


def evidence(ev, nod, cat, val):
    ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(nod)) \
        .with_evidence(cat, val) \
        .build()
    join_tree.set_observation(ev)


# padrão
print('\nCenário Padrão')
print_probs()

# cenario 1
evidence('evidence1', 'C9am', moreThan4,            0.5)
# evidence('evidence2', 'C3pm', moreThan4,            40)
# evidence('evidence3', 'H9am', moreThan60,           40)
# evidence('evidence4', 'H3pm', moreThan60,           40)
# evidence('evidence6', 'Rainfall', moreThan15,       40)
# evidence('evidence7', 'Sunshine', moreThan6,        40)
# evidence('evidence8', 'Temp9am', moreThan20,        40)
# evidence('evidence9', 'Temp3pm', moreThan20,        40)
# evidence('evidence10', 'WindGustSpeed', moreThan40, 40)

print('\nCenário #1')
print_probs()
