from calc_probabilities import Probability
from csv_reader import get_data_frame  # for data manipulation
import strings
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.pptc.inferencecontroller import InferenceController


dataFrame = get_data_frame('./dataset/weatherAUS-small.csv')
probability = Probability()

# Create nodes by using our earlier function to automatically calculate probabilities
C9am = BbnNode(Variable(0, 'C9am', [strings.lessOrEqualThan4, strings.moreThan4]), probability.calculate(
    dataFrame, child='Cloud9amCat', parent1='WindGustSpeedCat'))
C3pm = BbnNode(Variable(1, 'C3pm', [strings.lessOrEqualThan4, strings.moreThan4]), probability.calculate(
    dataFrame, child='Cloud3pmCat', parent1='WindGustSpeedCat', parent2='Cloud9amCat'))

H9am = BbnNode(Variable(2, 'H9am', [strings.lessOrEqualThan60, strings.moreThan60]), probability.calculate(
    dataFrame, child='Humidity9amCat', parent1='RainfallCat', parent2='Temp9amCat'))
H3pm = BbnNode(Variable(3, 'H3pm', [strings.lessOrEqualThan60, strings.moreThan60]), probability.calculate(
    dataFrame, child='Humidity3pmCat', parent1='Humidity9amCat', parent2='Temp3pmCat'))
RainToday = BbnNode(Variable(4, 'RainToday', [strings.no, strings.yes]), probability.calculate(
    dataFrame, child='RainToday'))
Rainfall = BbnNode(Variable(5, 'Rainfall', [strings.lessOrEqualThan15, strings.moreThan15]), probability.calculate(
    dataFrame, child='RainfallCat', parent1='RainToday'))
Sunshine = BbnNode(Variable(6, 'Sunshine', [strings.lessOrEqualThan6, strings.moreThan6]), probability.calculate(
    dataFrame, child='SunshineCat'))
Temp9am = BbnNode(Variable(7, 'Temp9am', [strings.lessOrEqualThan20, strings.moreThan20]), probability.calculate(
    dataFrame, child='Temp9amCat', parent1='SunshineCat'))
Temp3pm = BbnNode(Variable(8, 'Temp3pm', [strings.lessOrEqualThan20, strings.moreThan20]), probability.calculate(
    dataFrame, child='Temp3pmCat', parent1='Temp9amCat'))
WindGustSpeed = BbnNode(Variable(9, 'WindGustSpeed', [strings.lessOrEqualThan40, strings.moreThan40]),
                        probability.calculate(dataFrame, child='WindGustSpeedCat'))
RainTomorrow = BbnNode(Variable(10, 'RainTomorrow', [strings.no, strings.yes]), probability.calculate(
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
        if(node.to_dict()['variable']['id'] == 10):
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
evidence('evidence1', 'C9am', strings.moreThan4,            0.5)
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
