# ConfigParser 是用来读取配置文件的包
import configparser
import os
import numpy as np
from subprocess import Popen, PIPE
from genetic.population import Population, Individual, DenseUnit, ResUnit, PoolUnit
import logging
import sys
import multiprocessing
import time


class StatusUpdateTool(object):
    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)
    
    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False
    
    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")


class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        '''
        通过文件检索现在的gen_no
        '''
        id_list = []
        for _, _, file_names in os.walk('./populations'):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    id_list.append(int(file_name[6:8]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)
    
    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = './populations/begin_%02d.json' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)
    
    @classmethod
    def load_population(cls, prefix, gen_no):
        '''
        读取 .txt 文件，解析字符串
        '''
        file_name = './populations/%s_%02d.json' % (prefix, np.min(gen_no))
        params = StatusUpdateTool.get_init_params()
        pop = Population(params, gen_no)

        f = open(file_name, 'r')
        # *** strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        info = json.load(f)

        individuals = info["populations"]
        for i in individuals:
            indi_no = i["indi_no"]
            indi = Individual(i, indi_no)
        # indi_no = indi_start_line[5:]
        # indi = Individual(params, indi_no)
        
        pop.individuals.append(indi)
        f.close()

        # load the fitness to the individuals who have been evaluated, only suitable for the first generation
        if gen_no == 0:
            after_file_path = './populations/after_%02d.json' % (gen_no)
            if os.path.exists(after_file_path):
                # 从after_文件中取出已训练好的accuracy数据
                fitness_map = {}
                f = open(after_file_path, 'r')
                info = json.load(f)
                for i in info["populations"]:
                    if "accuracy" in i:
                        fitness_map[ i["indi_no"] ] = i["accuracy"]
                f.close()

                for indi in pop.individuals:
                    if indi.id in fitness_map:
                        indi.acc = fitness_map[indi.id]
        return pop

    @classmethod
    def read_template(cls):
        _path = './template/ls_evolution_net.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  #skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part1))

        line = f.readline().rstrip()  #skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part2))

        line = f.readline().rstrip()  #skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        #print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi):
        '''
        生成py文件的内容
        '''
        unit_list, forward_list = indi.file_string()
        
        part1, part2, part3 = cls.read_template()
        # 在文件头做时间标记
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')

        # 按顺序添加语句
        _str.extend(part1)
        _str.append('\n        %s' % ('# conv and bn_relu layers'))
        for s in unit_list:
            _str.append('        %s' % (s))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        
        _str.extend(part3)

        # 创建py文件
        file_name = './scripts/%s.py' % (indi.id)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()