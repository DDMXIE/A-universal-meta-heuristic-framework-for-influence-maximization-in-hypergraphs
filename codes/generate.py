import codecs
import numpy as np


def main():
    fout = codecs.open('run_command.txt', 'a', "utf-8")

    # filename_list = ['Algebra', 'Geometry', 'Restaurants-Rev', 'iAF1260b', 'iJO1366']
    filename_list = ['Algebra', 'Geometry', 'Restaurants-Rev', 'iAF1260b', 'iJO1366', 'Music-Rev', 'Bars-Rev', 'NDC-classes-unique-hyperedges']
    for net in filename_list:
        command = 'nohup python3 hyper_algorithm_hga.py %s >> nohup.out 2>&1 &' % net
        fout.write(command + '\n\n')
    fout.close()


if __name__ == "__main__":
    main()