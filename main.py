import strings
import nodes
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.pptc.inferencecontroller import InferenceController


def create_network(nodes):
    # Create Network
    return Bbn()\
        .add_node(nodes.C9am) \
        .add_node(nodes.C3pm) \
        .add_node(nodes.H9am) \
        .add_node(nodes.H3pm) \
        .add_node(nodes.RainToday) \
        .add_node(nodes.Rainfall) \
        .add_node(nodes.Sunshine) \
        .add_node(nodes.Temp9am) \
        .add_node(nodes.Temp3pm) \
        .add_node(nodes.RainTomorrow) \
        .add_node(nodes.WindGustSpeed) \
        .add_edge(Edge(nodes.RainToday, nodes.Rainfall, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.Rainfall, nodes.H9am, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.H9am, nodes.H3pm, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.H3pm, nodes.RainTomorrow, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.Sunshine, nodes.Temp9am, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.Temp9am, nodes.Temp3pm, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.Temp9am, nodes.H9am, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.Temp3pm, nodes.H3pm, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.C9am, nodes.C3pm, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.C3pm, nodes.RainTomorrow, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.WindGustSpeed, nodes.C9am, EdgeType.DIRECTED)) \
        .add_edge(Edge(nodes.WindGustSpeed, nodes.C3pm, EdgeType.DIRECTED)) \


    # Convert the BBN to a join tree
network = create_network(nodes)
join_tree = InferenceController.apply(network)

# Define a function for printing marginal probabilities


def show_probabilities():
    for node in join_tree.get_bbn_nodes():
        if(node.to_dict()['variable']['id'] == 10):
            potential = join_tree.get_bbn_potential(node)
            print("Probabilities for RainTomorrow:")
            print(potential)


def create_evidence(ev, nod, cat, val):
    ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(nod)) \
        .with_evidence(cat, val) \
        .build()
    join_tree.set_observation(ev)


# padrão
print('\nCenário Padrão')
show_probabilities()

# cenario 1
create_evidence('evidence1', 'C9am', strings.moreThan4, 0)
print('\nCenário #1')
show_probabilities()
