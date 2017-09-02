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

import importlib

import MLC.Log.log as lg
import numpy as np
import sys

from MLC.Common.LispTreeExpr.TreeNodes import TreeNode
from MLC.Common.LispTreeExpr.TreeNodes import LeafNode
from MLC.Common.LispTreeExpr.TreeNodes import InternalNode
from MLC.mlc_parameters.mlc_parameters import Config
from MLC.Common.Operations import Operations


def process_float(arg):
    str_arg = ("%." + Config.get_instance().get('POPULATION', 'precision') + "f") % (arg)
    return str_arg


def execute_op_without_warnings(op, log_prefix, exception_msg, arg1, arg2=None):
    result = None
    np.seterr(all='ignore')
    if arg2 is None:
        result = op(arg1)
    else:
        result = op(arg1, arg2)
    np.seterr(all='raise')
    lg.logger_.warn("{0}{1}".format(log_prefix, exception_msg))
    return result


class PlusNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "+", 1)

    def formal(self):
        return "(" + self._nodes[0].formal() + " + " + self._nodes[1].formal() + ")"

    def op_simplify(self):
        # If one of the arguments is zero, avoid the operation
        if self._node_arg_x_is_y(0, 0):
            return LeafNode(self._node_id, self._nodes[1].to_string())
        elif self._node_arg_x_is_y(1, 0):
            return LeafNode(self._node_id, self._nodes[0].to_string())

        # Non of the arguments are zero. Make the operation if they are not sensors
        if not self._nodes[0].is_sensor() and not self._nodes[1].is_sensor():
            arg = float(self._nodes[0].to_string()) + float(self._nodes[1].to_string())
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return arg_list[0] + arg_list[1]
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.add,
                                               arg1=arg_list[0],
                                               arg2=arg_list[1],
                                               log_prefix="[PLUS_NODE] Error: ",
                                               exception_msg=err)


class MinusNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "-", 1)

    def formal(self):
        return "(" + self._nodes[0].formal() + " - " + self._nodes[1].formal() + ")"

    def op_simplify(self):
        # if both arguments are equals, return 0
        if self._nodes[0].to_string() == self._nodes[1].to_string():
            return LeafNode(self._node_id, process_float(0))

        # If the second argument is zero, avoid the operation.
        if self._node_arg_x_is_y(1, 0):
            return LeafNode(self._node_id, self._nodes[0].to_string())

        # Non of the arguments are zero. Make the operation if they are not sensors
        if not self._nodes[0].is_sensor() and not self._nodes[1].is_sensor():
            arg = float(self._nodes[0].to_string()) - float(self._nodes[1].to_string())
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return arg_list[0] - arg_list[1]
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.subtract,
                                               arg1=arg_list[0],
                                               arg2=arg_list[1],
                                               log_prefix="[MINUS_NODE] Error: ",
                                               exception_msg=err)


class MultNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "*", 1)

    def formal(self):
        return "(" + self._nodes[0].formal() + " .* " + self._nodes[1].formal() + ")"

    def op_simplify(self):
        # If one or both of the arguments are zero, return zero
        if self._node_arg_x_is_y(0, 0) or self._node_arg_x_is_y(1, 0):
            return LeafNode(self._node_id, process_float(0))

        # If one of the arguments is zero, avoid the operation
        if self._node_arg_x_is_y(0, 1):
            return LeafNode(self._node_id, self._nodes[1].to_string())
        elif self._node_arg_x_is_y(1, 1):
            return LeafNode(self._node_id, self._nodes[0].to_string())

        if not self._nodes[0].is_sensor() and not self._nodes[1].is_sensor():
            arg = float(self._nodes[0].to_string()) * float(self._nodes[1].to_string())
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return arg_list[0] * arg_list[1]
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.multiply,
                                               arg1=arg_list[0],
                                               arg2=arg_list[1],
                                               log_prefix="[MULTI_NODE] Error: ",
                                               exception_msg=err)


class DivisionNode(InternalNode):
    PROTECTION = 0.001
    SIMPLIFY_PROTECTION = 0.01

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "/", 1)

    def formal(self):
        return "(my_div(" + self._nodes[0].formal() + "," + self._nodes[1].formal() + "))"

    def _process_division(self, dividend, divisor):
        if type(divisor) == np.ndarray:
            new_divisor = [DivisionNode.PROTECTION if np.abs(x) < DivisionNode.PROTECTION else np.abs(x) for x in divisor]
            return np.sign(divisor) * dividend / np.asarray(new_divisor)
        else:
            if abs(divisor) < DivisionNode.PROTECTION:
                return np.sign(divisor) * dividend / DivisionNode.PROTECTION

        return dividend / divisor

    def op_simplify(self):
        # If the first argument is zero, return zero
        if self._node_arg_x_is_y(0, 0):
            return LeafNode(self._node_id, process_float(0))

        # If the second argument is one, return the first argument
        if self._node_arg_x_is_y(1, 1):
            return LeafNode(self._node_id, self._nodes[0].to_string())

        if not self._nodes[0].is_sensor() and not self._nodes[1].is_sensor():
            # FIXME: Harcoded number. Change it
            if abs(float(self._nodes[1].to_string())) < DivisionNode.SIMPLIFY_PROTECTION:
                return LeafNode(self._node_id, process_float(0))
            else:
                arg = float(self._nodes[0].to_string()) / float(self._nodes[1].to_string())
                return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return self._process_division(arg_list[0], arg_list[1])
        except FloatingPointError as err:
            return execute_op_without_warnings(op=self._process_division,
                                               arg1=arg_list[0],
                                               arg2=arg_list[1],
                                               log_prefix="[DIV_NODE] Error: ",
                                               exception_msg=err)


class SineNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "sin", 3)

    def formal(self):
        return "sin(" + self._nodes[0].formal() + ")"

    def op_simplify(self):
        if not self._nodes[0].is_sensor():
            arg = np.sin(float(self._nodes[0].to_string()))
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return np.sin(arg_list[0])
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.sin,
                                               arg1=arg_list[0],
                                               log_prefix="[SIN_NODE] Error: ",
                                               exception_msg=err)


class CosineNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "cos", 3)

    def formal(self):
        return "cos(" + self._nodes[0].formal() + ")"

    def op_simplify(self):
        if not self._nodes[0].is_sensor():
            arg = np.cos(float(self._nodes[0].to_string()))
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return np.cos(arg_list[0])
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.cos,
                                               arg1=arg_list[0],
                                               log_prefix="[COS_NODE] Error: ",
                                               exception_msg=err)


class LogarithmNode(InternalNode):
    PROTECTION = 0.00001
    SIMPLIFY_PROTECTION = 0.01

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "log", 5)

    def formal(self):
        return "my_log(" + self._nodes[0].formal() + ")"

    def _process_arg(self, arg):
        if type(arg) == np.ndarray:
            return [LogarithmNode.PROTECTION if np.abs(x) < LogarithmNode.PROTECTION else np.abs(x) for x in arg]
        else:
            if abs(arg) < LogarithmNode.PROTECTION:
                return LogarithmNode.PROTECTION

        return abs(arg)

    def op_simplify(self):
        if not self._nodes[0].is_sensor():
            if float(self._nodes[0].to_string()) < LogarithmNode.SIMPLIFY_PROTECTION:
                arg = np.log(LogarithmNode.SIMPLIFY_PROTECTION)
            else:
                arg = np.log(float(self._nodes[0].to_string()))

            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return np.log(self._process_arg(arg_list[0]))
        except FloatingPointError as err:
            return execute_op_without_warnings(op=lambda x: np.log(self._process_arg(x)),
                                               arg1=arg_list[0],
                                               log_prefix="[LOG_NODE] Error: ",
                                               exception_msg=err)


class ExponentialNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "exp", 5)

    def formal(self):
        return "exp(" + self._nodes[0].formal() + ")"

    def op_simplify(self):
        if not self._nodes[0].is_sensor():
            lg.logger_.debug("[EXP_NODE] Value: " + self._nodes[0].to_string())
            try:
                arg = np.exp(float(self._nodes[0].to_string()))
            except OverflowError:
                # FIXME: See what to do with this expression, because there are problems when
                # an infinite value is the argument of a sinusoidal function
                return LeafNode(self._node_id, process_float(float("inf")))

            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return np.exp(arg_list[0])
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.exp,
                                               arg1=arg_list[0],
                                               log_prefix="[EXP_NODE] Error: ",
                                               exception_msg=err)


class TanhNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "tanh", 5)

    def formal(self):
        return "tanh(" + self._nodes[0].formal() + ")"

    def op_simplify(self):
        if not self._nodes[0].is_sensor():
            arg = np.tanh(float(self._nodes[0].to_string()))
            return LeafNode(self._node_id, process_float(arg))
        else:
            return self

    def op_compute(self, arg_list):
        try:
            return np.tanh(arg_list[0])
        except FloatingPointError as err:
            return execute_op_without_warnings(op=np.exp,
                                               arg1=arg_list[0],
                                               log_prefix="[TANH_NODE] Error: ",
                                               exception_msg=err)


class RootNode(InternalNode):

    def __init__(self, node_id):
        InternalNode.__init__(self, node_id, "", 0)

    def to_string(self):
        return " ".join([n.to_string() for n in self._nodes])

    def simplify(self):
        self._nodes = [node.simplify() for node in self._nodes]
        return self

    def compute(self):
        np.seterr(all='raise')
        if len(self._nodes) == 1:
            result = self._nodes[0].compute()
            np.seterr(all='warn')
            return result

        result = [node.compute() for node in self._nodes]
        np.seterr(all='warn')
        return result

    def formal(self):
        if len(self._nodes) == 1:
            return self._nodes[0].formal()

        return [n.formal() for n in self._nodes]

    def accept(self, visitor):
        for node in self._nodes:
            node.accept(visitor)


class OpNodeFactory:

    @staticmethod
    def make(op, node_id):
        if op == 'root':
            return RootNode(node_id)

        # instatiate node from string
        operation = Operations.get_instance().get_operation_from_op_string(op)
        node_module_name = ".".join(operation['tree_node_class'].split('.')[:-1])
        node_class_name = ".".join(operation['tree_node_class'].split('.')[-1:])

        node_module = importlib.import_module(node_module_name)
        node_class = getattr(node_module, node_class_name)

        return node_class(node_id)
