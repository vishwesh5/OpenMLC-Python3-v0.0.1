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

from PyQt5.QtChart import QChart, QChartView, QLineSeries, QSplineSeries, QScatterSeries
from PyQt5.QtChart import QValueAxis
from PyQt5.QtChart import QLogValueAxis
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF, QPainter

import numpy as np


class QtChartWrapper():
    #FIXME: Do a factory method or a decorator to make a line o spline series
    def __init__(self, show_legend=False, no_spline=False):
        self._ncurves = 0
        self._chart = QChart()

        if not show_legend:
            self._chart.legend().hide()

        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setUpdatesEnabled(True)

        self._xaxis = QValueAxis()
        self._yaxis = QValueAxis()

        self._no_spline = no_spline

        # Save the curves in order to add data to them later
        self._curves = []

    def remove_all_curves(self):
        self._chart.removeAllSeries()
        self._curves = []

    def set_title(self, title, font=None):
        self._chart.setTitle(title)

        if font != None:
            self._chart.setTitleFont(font)

    def set_object_name(self, name):
        self._view.setObjectName(name)

    def set_xaxis(self, log=False, label="", label_format="", tick_count=None):
        self._xaxis = self._create_axis(log, label, label_format, tick_count)
        self._chart.addAxis(self._xaxis, Qt.AlignBottom)

    def set_yaxis(self, log=False, label="", label_format="", tick_count=None):
        self._yaxis = self._create_axis(log, label, label_format, tick_count)
        self._chart.addAxis(self._yaxis, Qt.AlignLeft)

    def _create_axis(self, log, label, label_format, tick_count):
        axis = QValueAxis() if log == False else QLogValueAxis()
        axis.setTitleText(label)
        axis.setLabelFormat(label_format)

        if tick_count and not log:
            axis.setTickCount(tick_count)

        return axis

    def add_data(self, xdata, ydata, line_width=.1, color=None):
        curve = None
        if self._no_spline:
           curve = QLineSeries()
        else:
           curve = QSplineSeries()
        # curve = QSplineSeries()
        pen = curve.pen()

        if color is not None:
            pen.setColor(color)

        pen.setWidthF(line_width)
        curve.setPen(pen)
        curve.setUseOpenGL(True)
        # curve.append(self._series_to_polyline(xdata, ydata))

        # Convert the xdata and ydata into a list of points. Add them to the curve
        points = []
        for i in xrange(0, len(xdata)):
            points.append(QPointF(xdata[i], ydata[i]))
        curve.append(points)

        self._chart.addSeries(curve)
        self._curves.append(curve)
        self._ncurves += 1

        curve.attachAxis(self._xaxis)
        curve.attachAxis(self._yaxis)
        return self._ncurves - 1

    def add_line_curve(self, line_width=.1, color=None, legend=None):
        curve = QLineSeries()
        curve.setName(legend)
        pen = curve.pen()

        if color is not None:
            pen.setColor(color)

        pen.setWidthF(line_width)
        curve.setPen(pen)
        curve.setUseOpenGL(True)

        self._chart.addSeries(curve)
        self._curves.append(curve)
        self._ncurves += 1

        curve.attachAxis(self._xaxis)
        curve.attachAxis(self._yaxis)
        return self._ncurves - 1

    def add_scatter(self, marker_size=1, color=None, legend=None):
        curve = QScatterSeries()
        curve.setName(legend)

        if color is not None:
            curve.setColor(color)

        curve.setMarkerSize(marker_size)
        curve.setUseOpenGL(True)

        self._chart.addSeries(curve)
        self._curves.append(curve)
        self._ncurves += 1

        curve.attachAxis(self._xaxis)
        curve.attachAxis(self._yaxis)
        return self._ncurves - 1

    def append_point(self, curve_number, x, y):
        # FIXME: Do something different to uniquely identify the curve.
        # The curve number it is not enough
        try:
            self._curves[curve_number].append(QPointF(x, y))
            return True
        except IndexError:
            return False

    def get_widget(self):
        """
        Return the widget to be added to the GUI
        """
        return self._view

    def get_legend(self):
        return self._chart.legend()

    def _series_to_polyline(self, xdata, ydata):
        """Convert series data to QPolygon(F) polyline

        This code is derived from PythonQwt's function named
        `qwt.plot_curve.series_to_polyline`"""
        size = len(xdata)
        polyline = QPolygonF(size)
        pointer = polyline.data()
        dtype, tinfo = np.float, np.finfo  # integers: = np.int, np.iinfo
        pointer.setsize(2 * polyline.size() * tinfo(dtype).dtype.itemsize)
        memory = np.frombuffer(pointer, dtype)
        memory[:(size - 1) * 2 + 1:2] = xdata
        memory[1:(size - 1) * 2 + 2:2] = ydata
        return polyline
