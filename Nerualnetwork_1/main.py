import numpy as np

# Создание активационной функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# создание тренировочных входных данных
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])

# создание правильных ответов на тренировочные входные данные
training_outputs = np.array([[0, 1, 1, 1]]).T

# рандомный выбор вессов
synaptic_weights = 2 * np.random.random((3, 1)) - 1

# выводим рандомные веса
print("рандомные веса:")
print(synaptic_weights)

# метод обратного распространения ошибки (обучение нейронной сети)
for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

    synaptic_weights += adjustments

# выводим результаты обучения
print("веса после обучения:")
print(synaptic_weights)

print("результат после обучения:")
print(outputs)

# создаём не стандартный случай(случай которого не было при обучении)
new_inputs = np.array([0, 1, 1]) # новая ситуация
outputs = sigmoid( np.dot( new_inputs, synaptic_weights))

# выводим результат нестандартного случая
print("Новая ситуация:")
print(outputs)