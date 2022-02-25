import csv
import numpy as np

def converter():
    path = '../dataset/winequality-red.csv'
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        train = []
        contador = 0
        for row in reader:
          
          if(contador != 0): 
            fixed_acidity = row[0]
            train.append(np.double(fixed_acidity))

            volatile_acidity = row[1]
            train.append(np.double(volatile_acidity))

            citric_acid = row[2]
            train.append(np.double(citric_acid))

            residual_sugar = row[3]
            train.append(np.double(residual_sugar))

            chlorides = row[4]
            train.append(np.double(chlorides))

            free_sulfur_dioxide = row[5]
            train.append(np.double(free_sulfur_dioxide))

            total_sulfur_dioxide = row[6]
            train.append(np.double(total_sulfur_dioxide))

            density = row[7]
            train.append(np.double(density))

            ph = row[8]
            train.append(np.double(ph))

            sulphates = row[9]
            train.append(np.double(sulphates))

            alcohol = row[10]
            train.append(np.double(alcohol))

            quality = row[11]
            train.append(np.double(quality))
          contador += 1
        print(train)
        return train


converter()
