import numpy as np
import pandas as pd
from transform import Transform
from Hyperspreading import Hyperspreading
from tqdm import tqdm
import copy
import random
import networkx as nx

OPTIMAL_CHROMOSOME = []

def getTotalAdj(df_hyper_matrix, N):
    # 获取节点度序列
    deg_list = []
    nodes_arr = np.arange(N)
    for node in nodes_arr:
        node_list = []
        edge_set = np.where(df_hyper_matrix.loc[node] == 1)[0]
        for edge in edge_set:
            node_list.extend(list(np.where(df_hyper_matrix[edge] == 1)[0]))
        node_set = np.unique(np.array(node_list))
        deg_list.append(len(list(node_set)) - 1)
    return np.array(deg_list)


def get_fitness_score_list(df_hyper_matrix, N, p):

    arr = spreading_dynamics(df_hyper_matrix)
    return np.array(arr)


def get_fitness_score_by_nodes(nodes_list, fitness_score_list):
    # 给定节点序列，得出节点的适应性分数序列
    temp_list = []
    for node in nodes_list:
        temp_list.append(fitness_score_list[int(node)])
    return temp_list


def spreading_dynamics(df_hyper_matrix):
    hs = Hyperspreading()
    R = 10
    node_len = len(df_hyper_matrix.index.values)
    scale_val = np.zeros(node_len)
    for r in tqdm(range(R), desc='loading'):
        for node in range(node_len):
            seeds = [node]
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_val[node] = scale_val[node] + scale

    arr = scale_val / R
    return arr


def initialization(N, POPULATION_SIZE, CHROMOSOME_SIZE):
    # 随机抽取x倍（此处为3）设定种群数量的种群
    population_matrix = []
    for _ in range(POPULATION_SIZE * 3):
        population_matrix.append(list(np.random.choice(np.arange(N), size=CHROMOSOME_SIZE)))
    return population_matrix


def update_fitness(fitness_score_list, chromosomes):
    # 物竞 - 更新适应性分数轮盘
    fitness_total_score_list = []
    for chro in chromosomes:
        count = 0
        for gene in chro:
            count = count + fitness_score_list[int(gene)]
        fitness_total_score_list.append(count)
    return np.array(fitness_total_score_list)


def selection(chromosomes, POPULATION_SIZE, fitness):
    # 天择 - 选择 x 个优秀的群组
    idx = np.random.choice(len(chromosomes), size=POPULATION_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    population_matrix = []
    for row_id in idx:
        population_matrix.append(chromosomes[row_id])
    return population_matrix


def mutation(child, MUTATION_PROB, CHROMOSOME_SIZE, N):
    # 重组过程中发生的突变效果，跳出局部最优解
    if np.random.rand() < MUTATION_PROB:
        gene_idx = np.random.randint(low=0, high=CHROMOSOME_SIZE)
        while 1:
            node = np.random.randint(low=0, high=N)
            if node not in child:
                break
        child[gene_idx] = node
    return child


def crossover_and_mutation(chromosomes, fitness_score_list, CHROMOSOME_SIZE, CROSS_OVER_PROB, MUTATION_PROB, N):
    temp_chromosomes = []
    # 交叉重组 产生优秀子代
    for father in chromosomes:

        # 将父亲的先遗传，待交叉
        child = father

        # 以一定的交叉概率进行交叉现象
        if np.random.rand() < CROSS_OVER_PROB:

            # 选择一个母亲
            mother = chromosomes[np.random.choice(np.arange(len(chromosomes)), size=1)[0]]
            father_fitness_list = get_fitness_score_by_nodes(father, fitness_score_list)
            mother_fitness_list = get_fitness_score_by_nodes(mother, fitness_score_list)
            df_father = pd.DataFrame([father, father_fitness_list])
            df_mother = pd.DataFrame([mother, mother_fitness_list])
            df = df_father.join(df_mother, lsuffix='_left', rsuffix='_right')
            df = pd.DataFrame(df.values)

            # 去重
            duplicated_check_array = df.iloc[0].duplicated()
            duplicated_col_idx = list(np.where(duplicated_check_array == True)[0])
            df.drop(df.columns[duplicated_col_idx], axis=1, inplace=True)

            # 降序选择
            df.sort_values(by=df.index.tolist(), axis=1)
            sort_df = df.sort_values(by=1, axis=1, ascending=False)

            # 交叉产生的子代
            child = np.array(sort_df.iloc[0])[:CHROMOSOME_SIZE]

        # 变异 交叉过程不是完全复制，也有可能变异，产生新配型
        new_child = mutation(child, MUTATION_PROB, CHROMOSOME_SIZE, N)
        temp_chromosomes.append(list(new_child))
    chromosomes.extend(list(temp_chromosomes))
    return chromosomes



def evolve(df_hyper_matrix, CHROMOSOME_SIZE, POPULATION_SIZE, CROSS_OVER_PROB, MUTATION_PROB, N, GENERATIONS_TIMES, p):
    # 初始随机选取种子节点种群
    chromosomes = initialization(N, POPULATION_SIZE, CHROMOSOME_SIZE)
    fitness_score_list = get_fitness_score_list(df_hyper_matrix, N, p)
    # 适应性函数 - 生存策略
    fitness = update_fitness(fitness_score_list, chromosomes)


    for g_time in range(GENERATIONS_TIMES):
        print('######################### GENERATIONS_TIMES ' + str(g_time)+ ' #############################')
        # 物竞 - 种群优胜劣汰
        chromosomes = selection(chromosomes, POPULATION_SIZE, fitness)
        # 天择 - 交叉 变异 产生优秀子代
        chromosomes = crossover_and_mutation(chromosomes, fitness_score_list, CHROMOSOME_SIZE, CROSS_OVER_PROB, MUTATION_PROB, N)
        print(chromosomes)
        # 重置适应性函数轮盘
        fitness = update_fitness(fitness_score_list, chromosomes)

    return chromosomes


def HGNA_main(POPULATION_SIZE, CHROMOSOME_SIZE, CROSS_OVER_PROB, MUTATION_PROB, GENERATIONS_TIMES, p, fileName):
    # 初始化
    # POPULATION_SIZE = 500       # 种群数量
    # CHROMOSOME_SIZE = 25         # 染色体数量
    # CROSS_OVER_PROB = 0.8       # 交叉过程概率
    # MUTATION_PROB = 0.005       # 变异概率
    # GENERATIONS_TIMES = 100  # 演化步数
    # p = 0.01
    tf = Transform()
    # 构造超图矩阵
    # fileName = 'Restaurants-Rev'
    df_hyper_matrix, N = tf.changeEdgeToMatrix('../datasets/' + fileName + '.txt')
    print(df_hyper_matrix)

    # 迭代，优化子代配型
    chromosomes = evolve(df_hyper_matrix, CHROMOSOME_SIZE, POPULATION_SIZE, CROSS_OVER_PROB, MUTATION_PROB, N,
                         GENERATIONS_TIMES, p)

    print('***', list(chromosomes[-1:][0]))
    return list(chromosomes[-1:][0])


# if __name__ == '__main__':
#     hs = Hyperspreading()
#     tf = Transform()
#     POPULATION_SIZE = 500  # 种群数量
#     CHROMOSOME_SIZE = 25  # 染色体数量
#     CROSS_OVER_PROB = 0.8  # 交叉过程概率
#     MUTATION_PROB = 0.005  # 变异概率
#     GENERATIONS_TIMES = 100  # 演化步数
#     p = 0.01
#
#     fileName = 'Restaurants-Rev'
#
#     HGNA_main(POPULATION_SIZE, CHROMOSOME_SIZE, CROSS_OVER_PROB, MUTATION_PROB, GENERATIONS_TIMES, p, fileName)






