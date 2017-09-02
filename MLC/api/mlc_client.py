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

import os
import sys
sys.path.append(os.path.abspath(".") + "/../..")

from mlc import MLC
import requests
import json

import argparse

class MLCClient(MLC):
    def __init__(self, hostname, port):
        self._hostname = hostname
        self._port = port
        self._url = "http://"+hostname+":"+str(port)

    def open_experiment(self, experiment_name):
        json_action = json.dumps({"action": "open"})
        response = requests.put(self._url + "/mlc/workspace/experiments/%s" % experiment_name, json=json_action)
        return json.loads(response.text)

    def close_experiment(self, experiment_name):
        json_action = json.dumps({"action": "close"})
        response = requests.put(self._url + "/mlc/workspace/experiments/%s" % experiment_name, json=json_action)
        return json.loads(response.text)

    def get_workspace_experiments(self):
        response = requests.get(self._url+"/mlc/workspace/experiments")
        return json.loads(response.text)

    def delete_experiment_from_workspace(self, experiment_name):
        response = requests.delete(self._url + "/mlc/workspace/experiments/%s" % experiment_name)
        return json.loads(response.text)

    def new_experiment(self, experiment_name, experiment_configuration):
        json_config = json.dumps(experiment_configuration)
        response = requests.post(self._url+"/mlc/workspace/experiments/%s" % experiment_name, json=json_config)
        return json.loads(response.text)

    def get_experiment_configuration(self, experiment_name):
        raise NotImplementedError("MLC::get_experiment_configuration not implemented")

    def set_experiment_configuration(self, experiment_name, configuration):
        raise NotImplementedError("MLC::set_experiment_configuration not implemented")

    def go(self, experiment_name, to_generation, from_generation=0):
        json_action = json.dumps({"action":          "go",
                                  "from_generation": from_generation,
                                  "to_generation":   to_generation})

        response = requests.put(self._url + "/mlc/workspace/experiments/%s" % experiment_name, json=json_action)
        return json.loads(response.text)

    def get_experiment_info(self, experiment_name):
        response = requests.get(self._url + "/mlc/workspace/experiments/%s" % experiment_name)
        return json.loads(response.text)

    def get_generation(self, experiment_name, generation_number):
        raise NotImplementedError("MLC::get_generation not implemented")

    def get_individuals(self, experiment_name):
        raise NotImplementedError("MLC::get_individuals not implemented")


def parse_arguments():
    parser = argparse.ArgumentParser(description='MLC Server (API REST)')

    parser.add_argument('-s', '--mlc-server-hostname', default="127.0.0.1",
                        type=str, help='MLC server hostname.')

    parser.add_argument('-p', '--mlc-server-port', default=5000,
                        type=int, help='MLC Server listening port.')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()

    print "Connecting with the mlc server at http://%s:%s" % (arguments.mlc_server_hostname, arguments.mlc_server_port)

    first_experiment_name = "first_experiment"
    second_experiment_name = "second_experiment"

    mlc_client = MLCClient(arguments.mlc_server_hostname, arguments.mlc_server_port)

    print "Remember to remove all files in the workspace before running this POC!!!!"

    print "Experiments in the workspace: %s" % mlc_client.get_workspace_experiments()

    print "Add experiment '%s' to the workspace" % first_experiment_name
    experiment_name = "first_experiment_name"
    experiment_configuration = {"POPULATION":
                                    {"size": "100",
                                     "sensors": "1",
                                     "sensor_spec": "false",
                                     "sensor_list": "1, 5, 2, 4"}
                                }

    print mlc_client.new_experiment(first_experiment_name, experiment_configuration)

    print "Experiments in the workspace: %s" % mlc_client.get_workspace_experiments()

    print "Try to create another experiment with the same name must fail"
    print mlc_client.new_experiment(first_experiment_name, experiment_configuration)

    print "Experiments in the workspace: %s" % mlc_client.get_workspace_experiments()

    print "Add experiment %s to the workspace" % second_experiment_name
    print mlc_client.new_experiment(second_experiment_name, experiment_configuration)

    print "Experiments in the workspace: %s" % mlc_client.get_workspace_experiments()

    print "Delete Experiment %s" % second_experiment_name
    print mlc_client.delete_experiment_from_workspace(second_experiment_name)

    print "Experiments in the workspace: %s" % mlc_client.get_workspace_experiments()

    print "Remember to remove all files in the workspace before running this POC!!!!"