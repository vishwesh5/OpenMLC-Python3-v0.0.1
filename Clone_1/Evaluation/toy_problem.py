# -*- coding: utf-8 -*-
# MLC (Machine Learning Control): A genetic algorithm library to solve chaotic problems
# Copyright (C) 2015-2017, Thomas Duriez (thomas.duriez@gmail.com)
# Copyright (C) 2015, Adrian Durán (adrianmdu@gmail.com)
# Copyright (C) 2015-2017, Ezequiel Torres Feyuk (ezequiel.torresfeyuk@gmail.com)
# Copyright (C) 2016-2017, Marco Germano Zbrun (marco.germano@intraway.com)
# Copyright (C) 2016-2017, Raúl Lopez Skuba (raulopez0@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

# -*- coding: utf-8 -*-

import numpy as np
import MLC.Log.log as lg
import matplotlib.pyplot as plt
import random
import sys
import time
import csv
# import pandas as pd

from MLC.arduino.protocol import ArduinoUserInterface
from MLC.mlc_parameters.mlc_parameters import Config
from PyQt5.QtCore import Qt


def individual_data(indiv):
    global g_data
    # ==============================================================================
    #     SAMPLES = 201
    #     x = np.linspace(-10.0, 10.0, num=SAMPLES)
    #     y = np.tanh(4*x)
    # ==============================================================================
    # ==============================================================================
    # My Code to import features from the building data
    # dataset = pd.read_csv('/home/etorres/harsh.csv', delimiter='\t')
    try:
        if g_data is None:
            pass
    except NameError:
        g_data = None
        with open('/home/htomar/dataset.csv', 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                if g_data is None:
                    g_data = [[] for x in xrange(len(row))]

                for index in xrange(len(row)):
                    g_data[index].append(float(row[index]))

    x0 = np.array(g_data[0])  # Time
    x1 = np.array(g_data[1])  # Temperature
    x2 = np.array(g_data[2])  # Wind
    x3 = np.array(g_data[3])  # Solar
    x4 = np.array(g_data[4])  # Humidity
    x5 = np.array(g_data[7])  # IsHoliday
    x6 = np.array(g_data[8])  # Day of the Week
    y = np.array(g_data[5])  # Whole Building Energy

    # print "x0: {}".format(type(x0))
    # print "x1: {}".format(type(x1))
    # print "x2: {}".format(type(x2))
    # print "x3: {}".format(type(x3))
    # print "x4: {}".format(type(x4))
    # print "x5: {}".format(type(x5))
    # print "x6: {}".format(type(x6))
    # print "x0: {}".format(x0)
    # print "x1: {}".format(x1)
    # print "x2: {}".format(x2)
    # print "x3: {}".format(x3)
    # print "x4: {}".format(x4)
    # print "x5: {}".format(x5)
    # print "x6: {}".format(x6)
    # print "x6: {}".format(x6)
    # print "y: {}".format(y)

    # ==============================================================================
    # ==============================================================================
    # DON'T NEED TO ADD NOISE
    #     config = Config.get_instance()
    #     artificial_noise = config.getint('EVALUATOR', 'artificialnoise')
    #     y_with_noise = y + [random.random() / 2 - 0.25 for _ in xrange(SAMPLES)] + artificial_noise * 500
    #
    # ==============================================================================
    # ==============================================================================
    #     if isinstance(indiv.get_formal(), str):
    #         formal = indiv.get_formal().replace('S0', 'x')
    #     else:
    #         # toy problem support for multiple controls
    #         formal = indiv.get_formal()[0].replace('S0', 'x')
    # ==============================================================================
    # ==============================================================================
    # My definition for formal
    # TODO: This could be wrong. Check this line first
    #         formal: matlab interpretable expression of the individual
    if isinstance(indiv.get_formal(), str):
        formal = indiv.get_formal().replace('S0',
                                            'x0')  # Replacing S0 with x after obtaining the interpretable expression
        formal = formal.replace('S1', 'x1')
        formal = formal.replace('S2', 'x2')
        formal = formal.replace('S3', 'x3')
        formal = formal.replace('S4', 'x4')
        formal = formal.replace('S5', 'x5')
        formal = formal.replace('S6', 'x6')
    else:
        # toy problem support for multiple controls
        formal = indiv.get_formal()[0].replace('S0', 'x0')  # Should all of them be [0]? Mostly not. And this can be compressed of course
        formal = formal.replace('S1', 'x1')
        formal = formal.replace('S2', 'x2')
        formal = formal.replace('S3', 'x3')
        formal = formal.replace('S4', 'x4')
        formal = formal.replace('S5', 'x5')
        formal = formal.replace('S6', 'x6')

    # ==============================================================================
    # Calculate J like the sum of the square difference of the
    # functions in every point
    lg.logger_.debug('[POP][TOY_PROBLEM] Individual Formal: ' + formal)
    b = indiv.get_tree().calculate_expression([x0, x1, x2, x3, x4, x5, x6])
    # print b
    # If the expression doesn't have the term 'x',
    # the eval returns a value (float) instead of an array.
    # In that case transform it to an array
    # ==============================================================================
    #     if type(b) == float:
    #         b = np.repeat(b, SAMPLES)
    #
    #     return x, y, y_with_noise, b
    # ==============================================================================
    return x0, x1, x2, x3, x4, x5, x6, y, b


def cost(indiv):
    # x, y, y_with_noise, b = individual_data(indiv)
    x0, x1, x2, x3, x4, x5, x6, y, b = individual_data(indiv)
    # Deactivate the numpy warnings, because this sum could raise an overflow
    # Runtime warning from time to time
    np.seterr(all='ignore')
    # print "b: {}".format(b)
    # print "y: {}".format(y)
    # print "b: {}".format(type(b))
    # print "y: {}".format(type(y))

    array_size = 1
    try:
        array_size = b.size
    except AttributeError:
        pass

    cost_value = float(np.sum((b - y) ** 2)) / array_size
    np.seterr(all='warn')

    return cost_value

# ==============================================================================


def show_best(index, generation, indiv, cost, block=True):
    #     #x, y, y_with_noise, b = individual_data(indiv)
    x0, x1, x2, x3, x4, x5, x6, y, b = individual_data(indiv)
#
    x = np.linspace(0, y.size-1, num=y.size)

    #mean_squared_error = np.sqrt((y - b)**2 / (1 + np.absolute(x**2)))
    mean_squared_error = y - b  # This is just mean error
    # Put figure window on top of all other windows
    fig = plt.figure()
    fig.canvas.manager.window.setWindowModality(Qt.ApplicationModal)
    fig.canvas.manager.window.setWindowTitle("Best Individual")

    formal = None
    if type(indiv.get_formal()) == list:
        formal = indiv.get_formal()[0]
    else:
        formal = indiv.get_formal()

    plt.rc('font', family='serif')
    plt.suptitle("Generation N#{0} - Individual N#{1}\n"
                 "Cost: {2}\n Formal: {3}".format(generation,
                                                  index,
                                                  cost,
                                                  formal),
                 fontsize=12)

    plt.subplot(2, 1, 1)
    line1, = plt.plot(x, y, color='r', linewidth=2, label='Curve without noise')
    line3, = plt.plot(x, b, color='k', linewidth=2, label='Control Law (Individual)')
    plt.ylabel('Functions', fontsize=12, fontweight='bold')
    plt.xlabel('Samples', fontsize=12, fontweight='bold')
    plt.legend(handles=[line1, line3], loc=2)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x, mean_squared_error, '*r')
    plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    plt.xlabel('Samples', fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.yscale('log')

    plt.show(block=block)


