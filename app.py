import math
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from serial_deserial import to_file, deserialization
from work_with_arr import add_2_vecs_comps
from datetime import datetime
import sys

# import clr  # Специальная библиотека С# + Python

# if clr.AddReference("Program"):  # ссылка на dll
#     from Brainy import *  # код из С#


TRESHOLD_FUNC = 0
TRESHOLD_FUNC_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11
INIT_W_CONST = 12
INIT_RANDN = 13
SOFTMAX = 14
SOFTMAX_DERIV = 15
PIECE_WISE_LINEAR = 16
PIECE_WISE_LINEAR_DERIV = 17
TRESHOLD_FUNC_HALF = 18
TRESHOLD_FUNC_HALF_DERIV = 19
MODIF_MSE = 20


ready = False

# Различные операции по числовому коду


def operations(op, x):
    global ready
    alpha_leaky_relu = 1.7159
    alpha_sigmoid = 2
    alpha_tan = 1.7159
    beta_tan = 2/3
    if op == RELU:
        if (x <= 0):
            return 0
        else:
            return x
    elif op == RELU_DERIV:
        if (x <= 0):
            return 0
        else:
            return 1
    elif op == TRESHOLD_FUNC:
        if (x > 0):
            return 1
        else:
            return 0
    elif op == TRESHOLD_FUNC_HALF:
        if x >= 1/2:
            return 1
        else:
            return 0
    elif op == TRESHOLD_FUNC_HALF_DERIV:
        return 1
    elif op == PIECE_WISE_LINEAR:
        if x >= 1/2:
            return 1
        elif x < 1/2 and x > -1/2:
            return x
        elif x <= -1/2:
            return 0
    elif op == PIECE_WISE_LINEAR_DERIV:
        return 1
    elif op == TRESHOLD_FUNC_DERIV:
        return 1
    elif op == LEAKY_RELU:
        if (x <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == LEAKY_RELU_DERIV:
        if (x <= 0):
            return alpha_leaky_relu
        else:
            return 1
    elif op == SIGMOID:
        y = 1 / (1 + math.exp(-alpha_sigmoid * x))
        return y
    elif op == SIGMOID_DERIV:
        y = 1 / (1 + math.exp(-alpha_sigmoid * x))
        return alpha_sigmoid * y * (1 - y)
    elif op == INIT_W_MY:
        if ready:
            ready = False
            return -0.567141530112327
        ready = True
        return 0.567141530112327
    elif op == INIT_W_RANDOM:

        return random.random()
    elif op == TAN:
        y = alpha_tan * math.tanh(beta_tan * x)
        return y
    elif op == TAN_DERIV:
        y = alpha_tan * math.tanh(beta_tan * x)
        return beta_tan / alpha_tan * (alpha_tan * alpha_tan - y * y)
    elif op == INIT_W_CONST:
        return 0.567141530112327
    elif op == INIT_RANDN:
        return np.random.randn()
    else:
        print("Op or function does not support ", op)


class Dense:
    def __init__(self):  # конструктор
        self.in_ = None  # количество входов слоя
        self.out_ = None  # количество выходов слоя
        self.matrix = [0] * 10  # матрица весов
        self.cost_signals = [0] * 10  # вектор взвешенного состояния нейронов
        self.act_func = RELU
        self.hidden = [0] * 10  # вектор после функции активации
        self.errors = [0] * 10  # вектор ошибок слоя
        self.batch_acc_tmp_l = [0] * 10
        self.with_bias = False
        for row in range(10):  # создаем матрицу весов
            # подготовка матрицы весов,внутренняя матрица
            self.inner_m = list([0] * 10)
            self.matrix[row] = self.inner_m


# class Nn_params:
#     net = [None] * 2  # Двойной перпецетрон
#     for l_ind in range(2):
#         net[l_ind] = Dense()
#     sp_d = -1  # алокатор для слоев
#     nl_count = 0  # количество слоев

#     # разные параметры
#     loss_func = MODIF_MSE
#     alpha_leaky_relu = 0.01
#     alpha_sigmoid = 0.42
#     alpha_tan = 1.7159
#     beta_tan = 2 / 3

################### Функции обучения ######################


class NetCon:
    def __init__(self):
        self.net = [None] * 2  # Двойной перпецетрон
        for l_ind in range(2):
            self.net[l_ind] = Dense()
        self.sp_d = -1  # алокатор для слоев
        self.nl_count = 0  # количество слоев

    def make_hidden(self, layer_ind, inputs: list):
        layer = self.net[layer_ind]
        for row in range(layer.out_):
            tmp_v = 0
            for elem in range(layer.in_):
                if layer.with_bias:
                    if elem == 0:
                        tmp_v += layer.matrix[row][elem] * 1
                    else:
                        tmp_v += layer.matrix[row][elem] * inputs[elem]

                else:
                    tmp_v += layer.matrix[row][elem] * inputs[elem]

            layer.cost_signals[row] = tmp_v
            val = operations(layer.act_func, tmp_v)
            layer.hidden[row] = val

    def get_hidden(self, objLay: Dense):
        return objLay.hidden

    def feed_forwarding(self, inputs):
        self.make_hidden(0, inputs)
        j = self.nl_count
        for i in range(1, j):
            inputs = self.get_hidden(self.net[i - 1])
            self.make_hidden(i, inputs)

        last_layer = self.net[j-1]

        return self.get_hidden(last_layer)

    def cr_lay(self,   in_=0, out_=0, act_func=None, with_bias=False, init_w=INIT_W_RANDOM):
        self.sp_d += 1
        layer = self.net[self.sp_d]
        layer.in_ = in_
        layer.out_ = out_
        layer.act_func = act_func

        if with_bias:
            layer.with_bias = True
        else:
            layer.with_bias = False

        if with_bias:
            in_ += 1
        for row in range(out_):
            for elem in range(in_):
                layer.matrix[row][elem] = operations(
                    init_w, 0)

        self.nl_count += 1

    def calc_out_error(self,  targets, samples_count, batch_size):
        layer = self.net[self.nl_count-1]
        out_ = layer.out_
        print("sample_count", samples_count)

        for row in range(out_):
            # накапливаем ошибку на выходе
            if samples_count % (batch_size + 1) != 0:
                layer.batch_acc_tmp_l[row] +=\
                    (layer.hidden[row] - targets[row]) * operations(
                    layer.act_func + 1, layer.hidden[row])
                samples_count += 1
            else:
                # применяем ошибку
                layer.errors[row] = layer.batch_acc_tmp_l[row]
                # layer.batch_acc_tmp_l[row]=0
                samples_count = 1

        return samples_count

    def calc_hid_error(self,  layer_ind, samples_count, batch_size):
        layer = self.net[layer_ind]
        layer_next = self.net[layer_ind + 1]
        for elem in range(layer.in_):
            summ = 0
            for row in range(layer.out_):
                if samples_count % batch_size == 0:
                    summ += layer_next.matrix[row][elem] * \
                        layer_next.errors[row]
            layer.errors[elem] = summ * operations(
                layer.act_func + 1, layer.hidden[elem])

    def upd_matrix(self, layer_ind, errors, inputs, lr):
        layer = self.net[layer_ind]
        for row in range(layer.out_):
            error = errors[row]
            for elem in range(layer.in_):
                if layer.with_bias:
                    if elem == 0:
                        layer.matrix[row][elem] -= lr * \
                            error * 1
                    else:
                        layer.matrix[row][elem] -= lr * \
                            error * inputs[elem]
                else:
                    layer.matrix[row][elem] -= lr * \
                        error * inputs[elem]

    def calc_diff(self, out_nn, teacher_answ):
        diff = [0] * len(out_nn)
        for row in range(len(teacher_answ)):
            diff[row] = out_nn[row] - teacher_answ[row]
        return diff

    def get_err(self, diff):
        sum = 0
        for row in range(len(diff)):
            sum += diff[row] * diff[row]
        return sum


#############################################


def plot_gr(_file: str, errors: list, epochs: list) -> None:
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    ax.plot(epochs, errors,
            label="learning",
            )
    plt.xlabel('Эпоха обучения')
    plt.ylabel('loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    plt.show()


# train_inp = ((1, 1), (0, 0), (0, 1), (1, 0))  # Логическое И
# train_out = ([1], [0], [0], [0])


train_inp = ((1, 0, 0, 0, 1, 0, 0, 0),
             (0, 1, 0, 0, 1, 0, 0, 0),
             (1, 0, 0, 0, 0, 0, 0, 1),
             (0, 1, 0, 0, 0, 0, 0, 1)
             )

train_out = ((1, 0, 1, 0),
             (1, 0, 0, 0),
             (1, 0, 1, 1),
             (1, 0, 0, 1))


def main():
    epochs = 500
    l_r = 0.01
    batch_size = 2
    samples_count = 1

    errors_y = []
    epochs_x = []

    # Создаем обьект параметров сети

    net = NetCon()
    # Создаем слои
    net.cr_lay(8, 4, TRESHOLD_FUNC_HALF, False, INIT_W_MY)
    # n = cr_lay( 3, 1, SIGMOID, True, INIT_W_MY)

    for ep in range(epochs):  # Кол-во повторений для обучения
        gl_e = 0
        for single_array_ind in range(len(train_inp)):

            inputs = train_inp[single_array_ind]
            output = net.feed_forwarding(inputs)

            # print("output", output)

            e = net.calc_diff(output, train_out[single_array_ind])

            gl_e += net.get_err(e)

            samples_count = net.calc_out_error(
                train_out[single_array_ind], samples_count, batch_size)

            # Обновление весов
            net.upd_matrix(0, net.net[0].errors, inputs,
                           l_r)

            # samples_count += 1

        gl_e /= 2
        print("error", gl_e)
        print("ep", ep)
        print()

        errors_y.append(gl_e)
        epochs_x.append(ep)

        if gl_e < 0.3:
            break

    plot_gr('gr.png', errors_y, epochs_x)

    for single_array_ind in range(len(train_inp)):
        inputs = train_inp[single_array_ind]

        output_2_layer = net.feed_forwarding(inputs)

        equal_flag = 0
        out_net = net.net[0].out_
        for row in range(out_net):
            elem_net = output_2_layer[row]
            elem_train_out = train_out[single_array_ind][row]
            if elem_net > 0.5:
                elem_net = 1
            else:
                elem_net = 0
            print("elem:", elem_net)
            print("elem tr out:", elem_train_out)
            if elem_net == elem_train_out:
                equal_flag = 1
            else:
                equal_flag = 0
                break
        if equal_flag == 1:
            print('-vecs are equal-')
        else:
            print('-vecs are not equal-')

        print("========")

    # to_file(nn_params, nn_params.net, loger, 'wei1.my')


main()
