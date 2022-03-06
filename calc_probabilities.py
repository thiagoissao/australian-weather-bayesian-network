import pandas as pd  # for data manipulation


class Probability:

    def calculate(self, data, child, parent1=None, parent2=None):
        if parent1 == None:
            crosstab = pd.crosstab(
                data[child], 'Empty', margins=False, normalize='columns')
            prob = self.__formatCrosstab(crosstab)
            return prob

        # with one or two parents
        return self.__calculate_with_parents(data, child, parent1, parent2)

    def __calculate_with_parents(self, data, child, parent1, parent2=None):
        valuesToGroup = self.__getValuesToGroupByInTheRows(
            data, parent1, parent2)
        crosstab = pd.crosstab(
            valuesToGroup, data[child], margins=False, normalize='index')
        prob = self.__formatCrosstab(crosstab)
        return prob

    def __getValuesToGroupByInTheRows(self, data, parent1, parent2=None):
        values = [data[parent1]]
        if(parent2 != None):
            values.append(data[parent2])
        return values

    def __formatCrosstab(self, crosstab):
        return crosstab.sort_index().to_numpy().reshape(-1).tolist()
