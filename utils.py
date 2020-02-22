# ConfigParser 是用来读取配置文件的包
import configparser
import os
import numpy as np
from subprocess import Popen, PIPE
from genetic.population import Population, Individual
import logging
import sys
import multiprocessing
import time

from collections import defaultdict, OrderedDict
import json
import copy


class StatusUpdateTool(object):
    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False
    
    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read('global.ini')
        config.set(section, key, value)
        config.write(open('global.ini', 'w'))

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def get_pop_size(cls):
        rs = cls.__read_ini_file('settings', 'pop_size')
        return int(rs)

    @classmethod
    def get_epoch_size(cls):
        rs = cls.__read_ini_file('network', 'epoch')
        return int(rs)
        
    @classmethod
    def get_learning_rate(cls):
        rs = cls.__read_ini_file('network', 'learning_rate')
        return float(rs)

    @classmethod
    def get_individual_max_length(cls):
        rs = cls.__read_ini_file('network', 'max_length')
        return int(rs)

    @classmethod
    def get_input_channel(cls):
        rs = cls.__read_ini_file('network', 'input_channel')
        return int(rs)

    @classmethod
    def get_output_channel(cls):
        rs = cls.__read_ini_file('network', 'output_channel')
        channels = []
        for i in rs.split(','):
            channels.append(int(i))
        return channels

    @classmethod
    def get_num_class(cls):
        rs = cls.__read_ini_file('network', 'num_class')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p

    @classmethod
    def generate_pops_begin(cls):
        '''
        将初始化种群结构写入文件
        '''
        model = defaultdict(list)

        l0 = defaultdict(list)
        # l0["edges_in"].append(None)
        l0["edges_out"].append(0)
        l0["type"] = "linear"
        l0["inputs_mutable"] = 0
        l0["outputs_mutable"] = 0
        l0["properties_mutable"] = 0

        l1 = defaultdict(list)
        l1["edges_in"].append(0)
        l1["edges_out"].append(1)
        l1["type"] = "linear"
        l1["inputs_mutable"] = 0
        l1["outputs_mutable"] = 0
        l1["properties_mutable"] = 0

        l2 = defaultdict(list)
        l2["edges_in"].append(1)
        # l2["edges_out"].append(None)
        l2["type"] = "Global Pooling"
        l2["inputs_mutable"] = 0
        l2["outputs_mutable"] = 0
        l2["properties_mutable"] = 0

        # vertices = defaultdict(list)
        model["vertices"].append(l0)
        model["vertices"].append(l1)
        model["vertices"].append(l2)

        edg1 = defaultdict(list)
        edg1["from_vertex"] = 0
        edg1["to_vertex"] = 1
        edg1["type"] = "identity"

        edg2 = defaultdict(list)
        edg2["from_vertex"] = 1
        edg2["to_vertex"] = 2
        edg2["type"] = "identity"

        # edges = defaultdict(list)
        model["edges"].append(edg1)
        model["edges"].append(edg2)

        pops_dict = {
            'version': "1.0",
            'explain': {
                'used': True,
                'details': "this is for initialize population",
            }
        }

        pops_dict['pop_size'] = cls.get_pop_size()
        pops_dict['max_len'] = cls.get_individual_max_length()
        pops_dict['image_channel'] = cls.get_input_channel()
        pops_dict['output_channel'] = cls.get_output_channel()
        pops_dict['num_class'] = cls.get_num_class()
        pops_dict['genetic_prob'] = cls.get_genetic_probability()

        pops_dict["populations"] = []

        # TODO: 应该要保证和Population.initialize中的pop_size相同
        for i in range(cls.get_pop_size()):
            t_model = copy.deepcopy(model)
            t_model["indi_no"] = i
            pops_dict["populations"].append(t_model)

        json_str = json.dumps(pops_dict)
        # TODO: 在并行系统中一定是"00"吗?
        with open('./populations/begin_00.json', 'w') as json_file:
            json_file.write(json_str)
        
        return pops_dict["populations"]

    @classmethod
    def get_init_params(cls):
        '''
        初始化种群参数，创建begin_00.json种群初始文件
        '''
        params = {}
        params['pop_size'] = cls.get_pop_size()
        params['max_len'] = cls.get_individual_max_length()
        params['image_channel'] = cls.get_input_channel()
        params['output_channel'] = cls.get_output_channel()

        params['num_class'] = cls.get_num_class()
        params['genetic_prob'] = cls.get_genetic_probability()

        params['epoch'] = cls.get_epoch_size()
        params['learning_rate'] = cls.get_learning_rate()

        # params['min_resnet'], params['max_resnet'] = cls.get_resnet_limit()
        # params['min_pool'], params['max_pool'] = cls.get_pool_limit()
        # params['min_densenet'], params['max_densenet'] = cls.get_densenet_limit()

        # params['min_resnet_unit'], params['max_resnet_unit'] = cls.get_resnet_unit_length_limit()

        # params['k_list'] = cls.get_densenet_k_list()
        # params['max_k12_input_channel'], params['min_k12'], params['max_k12'] = cls.get_densenet_k12()
        # params['max_k20_input_channel'], params['min_k20'], params['max_k20'] = cls.get_densenet_k20()
        # params['max_k40_input_channel'], params['min_k40'], params['max_k40'] = cls.get_densenet_k40()
        
        # TODO: 创建begin_00种群初始文件(每次都要新创建吗?)
        params['populations'] = cls.generate_pops_begin()

        return params

class Log(object):
    _logger = None
    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EvoCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)
    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)

class GPUTools(object):
    @classmethod
    def _get_available_gpu_plain_info(cls):
        gpu_info_list = []
        #read the information
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        for line_no in range(len(lines)-3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())
        #parse the information
        print(gpu_info_list)
        if len(gpu_info_list) == 1:
            if gpu_info_list[0].startswith('No'): #GPU outputs: No running processes found
                return 10000 # indicating all the gpus are available
            else:
                info_array = gpu_info_list[0].split(' ', 1)
                if info_array[0] == '0':
                    Log.info('GPU_QUERY-GPU#1 and # are available, choose GPU#1')
                    return 1
                elif info_array[0] == '1':
                    Log.info('GPU_QUERY-GPU#0 and #2 is available, choose GPU#2')
                    return 2
                else:
                    Log.info('GPU_QUERY-GPU#0 and #1 is available, choose GPU#0')
                    return 0

        elif len(gpu_info_list) == 2:
            info_array1 = gpu_info_list[0].split(' ', 1)
            info_array2 = gpu_info_list[1].split(' ', 1)
            gpu_use_list = [info_array1[0], info_array2[0]]
            if '0' not in gpu_use_list:
                Log.info('GPU_QUERY-GPU#0 is available')
                return 0
            if '1' not in gpu_use_list:
                Log.info('GPU_QUERY-GPU#1 is available')
                return 1
            if '2' not in gpu_use_list:
                Log.info('GPU_QUERY-GPU#2 is available')
                return 2
        else:
            Log.info('GPU_QUERY-No available GPU')
            return None

    @classmethod
    def all_gpu_available(cls):
        plain_info = cls._get_available_gpu_plain_info()
        if plain_info is not None and plain_info == 10000:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            return False

    @classmethod
    def detect_availabel_gpu_id(cls):
        plain_info = cls._get_available_gpu_plain_info()
        if plain_info is None:
            return None
        elif plain_info == 10000:
            Log.info('GPU_QUERY-None of the GPU is occupied, return the first one')
            Log.info('GPU_QUERY-GPU#0 is available')
            return 0
        else:
            return plain_info

'''
class CPUTools(object):
    @classmethod
    def _get_available_cpu_plain_info(cls):
        cpu_info_list = []
        #read the information
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        for line_no in range(len(lines)-3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                cpu_info_list.append(lines[line_no][1:-1].strip())
        #parse the information
        print(cpu_info_list)
        if len(cpu_info_list) == 1:
            if cpu_info_list[0].startswith('No'): #CPU outputs: No running processes found
                return 10000 # indicating all the cpus are available
            else:
                info_array = cpu_info_list[0].split(' ', 1)
                if info_array[0] == '0':
                    Log.info('CPU_QUERY-CPU#1 and # are available, choose CPU#1')
                    return 1
                elif info_array[0] == '1':
                    Log.info('CPU_QUERY-CPU#0 and #2 is available, choose CPU#2')
                    return 2
                else:
                    Log.info('CPU_QUERY-CPU#0 and #1 is available, choose CPU#0')
                    return 0

        elif len(cpu_info_list) == 2:
            info_array1 = cpu_info_list[0].split(' ', 1)
            info_array2 = cpu_info_list[1].split(' ', 1)
            cpu_use_list = [info_array1[0], info_array2[0]]
            if '0' not in cpu_use_list:
                Log.info('CPU_QUERY-CPU#0 is available')
                return 0
            if '1' not in cpu_use_list:
                Log.info('CPU_QUERY-CPU#1 is available')
                return 1
            if '2' not in cpu_use_list:
                Log.info('CPU_QUERY-CPU#2 is available')
                return 2
        else:
            Log.info('CPU_QUERY-No available CPU')
            return None

    @classmethod
    def all_cpu_available(cls):
        plain_info = cls._get_available_cpu_plain_info()
        if plain_info is not None and plain_info == 10000:
            Log.info('CPU_QUERY-None of the CPU is occupied')
            return True
        else:
            return False

    @classmethod
    def detect_availabel_cpu_id(cls):
        plain_info = cls._get_available_cpu_plain_info()
        if plain_info is None:
            return None
        elif plain_info == 10000:
            Log.info('CPU_QUERY-None of the CPU is occupied, return the first one')
            Log.info('CPU_QUERY-CPU#0 is available')
            return 0
        else:
            return plain_info
'''


class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def load_cache_data(cls):
        '''
        读取_key(卷积网络结构的hash映射)、与_acc(准确率)
        '''
        file_name = './populations/cache.json'
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            # for each_line in f:
            #     rs_ = each_line.strip().split(';')
            #     _map[rs_[0]] = '%.5f' % (float(rs_[1]))
            info = json.load(f)
            f.close()
        # return _map
        return info["cache"]

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        '''
        由FitnessEvaluate.evaluate调用, 种群全部训练完成后, 将个体结构存储起来, 防止出现同结构重复训练
        TODO: 目前使用列表记录，应可使用映射
        '''
        _map = cls.load_cache_data()
        for indi in individuals:
            _key,_str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                f = open('./populations/cache.json', 'r')
                info = json.load(f)

                item = defaultdict(list)
                item["key"] = _key
                item["accuracy"] = _acc
                item["string"] = _str

                info["cache"].append(item)
                json_str = json.dumps(info)
                with open('./populations/cache.json', 'w') as json_file:
                    json_file.write(json_str)
                
                # _map[_key] = _acc
            
            # if _key not in _map:
            #     Log.info('Add record into cache, id:%s, acc:%.5f'%(_key, _acc))
            #     f = open('./populations/cache.txt', 'a+')
            #     _str = '%s;%.5f;%s\n'%(_key, _acc, _str)
            #     f.write(_str)
            #     f.close()
            #     _map[_key] = _acc

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
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = './populations/mutation_%02d.json'%(gen_no)
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

        # individuals = info["populations"]
        for i in info["populations"]:
            individual_item = info
            individual_item["net"] = i
            indi_no = i["indi_no"]
            indi = Individual(individual_item, indi_no)
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

        f = open(_path, 'r', encoding='UTF-8')
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
        script_file_handler = open(file_name, 'w', encoding='UTF-8')
        script_file_handler.write('\n'.join(_str))
        # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
        script_file_handler.flush()
        script_file_handler.close()
    
    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()