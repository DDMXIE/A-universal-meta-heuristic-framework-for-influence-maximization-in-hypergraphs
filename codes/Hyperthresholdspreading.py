import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from fractions import Fraction
import matplotlib.pyplot as plt

class Hyperthresholdspreading:

    def compute_pair_weight(self, matrix):
        np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
        neighbors_matrix = np.dot(matrix, matrix.T)
        neighbors_matrix[np.eye(len(neighbors_matrix), dtype=np.bool_)] = 0
        neighbors_matrix = np.array(neighbors_matrix).astype(float)
        for row_index in range(len(neighbors_matrix)):
            total_weights = np.sum(neighbors_matrix[row_index])
            if total_weights != 0:
                neighbors_matrix[row_index] = neighbors_matrix[row_index] / np.sum(neighbors_matrix[row_index])
        return neighbors_matrix


    def format_neighbors_matrix(self, neighbors_matrix):
        neighbors_dict = {}
        neighbors_value_dict = {}
        for row_idx in range(len(neighbors_matrix)):
            row = neighbors_matrix[row_idx]
            neighbors_dict[row_idx] = []
            for item_idx in range(len(list(row))):
                if row[item_idx] > 0:
                    neighbors_dict[row_idx].append(item_idx)
                    neighbors_value_dict[(row_idx, item_idx)] = row[item_idx]
        return neighbors_dict, neighbors_value_dict

    # def average_results(self, incidence_matrix, neighbors_dict, neighbors_value_dict, T, node_threshold):
    #     temporal_infected_nodes_matrix = []
    #     for _ in tqdm(range(10), desc="Loading..."):
    #         for init_seed in range(len(incidence_matrix)):
    #             temporal_infected_nodes_list = hyper_threshold_spreading(incidence_matrix, neighbors_dict,
    #                                                                      neighbors_value_dict, T,
    #                                                                      node_threshold, init_seed)
    #             temporal_infected_nodes_matrix.append(temporal_infected_nodes_list)
    #     return pd.DataFrame(temporal_infected_nodes_matrix).mean(axis=0)

    def hyper_threshold_spreading(self, incidence_matrix, neighbors_dict, neighbors_value_dict,
                                  inital_activated_seed):
        T = 5
        activated_nodes_list = inital_activated_seed
        list_total = np.arange(len(incidence_matrix))
        temporal_infected_nodes_list = [1]
        node_rdn_threshold = np.random.random(size=len(list(neighbors_dict.keys())))
        for t in range(1, T + 1):
            temp_activated_node_list = []
            # 对于网络中不在激活节点中的任一节点
            non_activated_set = set(list_total) - set(activated_nodes_list)
            for node in non_activated_set:
                # 找到该节点所有的前驱节点（相邻节点）
                pre_nodes_list = neighbors_dict[node]
                total_threshold = 0
                # 计算所有前驱节点在上一步激活节点中的阈值之和
                pre_and_act_nodes = list(set(pre_nodes_list) & set(activated_nodes_list))
                # total_threshold = len(pre_and_act_nodes) / len(pre_nodes_list)
                for pre_node in pre_and_act_nodes:
                    total_threshold = total_threshold + neighbors_value_dict[(node, pre_node)]
                # if total_threshold >= random.random():
                if total_threshold >= node_rdn_threshold[node]:
                    # node_threshold = 0.21
                    temp_activated_node_list.append(node)
            activated_nodes_list.extend(np.unique(temp_activated_node_list))
            temporal_infected_nodes_list.append(len(activated_nodes_list))
        return temporal_infected_nodes_list[-1:][0], activated_nodes_list