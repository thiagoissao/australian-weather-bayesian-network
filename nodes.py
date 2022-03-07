from calc_probabilities import Probability
from csv_reader import get_data_frame
import strings
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable


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
