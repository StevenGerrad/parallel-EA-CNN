from utils import Utils, GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep

from collections import defaultdict, OrderedDict
import json


class FitnessEvaluate(object):
    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        '''
        生成py执行脚本
        '''
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        """
        load fitness from cache file
        """
        # 从cache文件中查找fitness数据，防止重复训练，其中_key为hash值
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f' %
                                (indi.id, _key, float(_acc), indi.acc))
                indi.acc = float(_acc)
        self.log.info('Total hit %d individuals for fitness' % (_count))

        # 查找是否有未训练完成的模型
        # p_l = []

        has_evaluated_offspring = False
        for indi in self.individuals:
            if indi.acc < 0:
                has_evaluated_offspring = True
                time.sleep(60)

                #############################################################################################
                # template change BELOW
                #############################################################################################

                # TODO:
                # cpu_id 得是一个数字
                cpu_id = 0
                file_name = indi.id
                module_name = 'scripts.%s' % (file_name)
                if module_name in sys.modules.keys():
                    self.log.info('Module:%s has been loaded, delete it' % (module_name))
                    del sys.modules[module_name]
                    # import_module: 根据字符串导入模块
                    # _module = importlib.import_module('..', module_name)
                    # _module = importlib.import_module('..', module_name)
                    _module = importlib.import_module(module_name)
                else:
                    # _module = importlib.import_module('..', module_name)
                    _module = importlib.import_module(module_name)
                
                # getattr() 函数用于返回一个对象属性值
                # _class = getattr(_module, 'TrainModel')
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()
                p = Process(target=cls_obj.do_work, args=('%d' % (cpu_id), file_name, indi.learning_rate, ))
                
                # p_l.append(p)
                p.start()
                # TODO: 内存不太够用
                # p.join()

                #############################################################################################
                # template change UP
                #############################################################################################
                
            else:
                file_name = indi.id
                self.log.info('%s has inherited the fitness as %.5f, no need to evaluate' %
                              (file_name, indi.acc))
                # f = open('./populations/after_%s.json' % (file_name[4:6]), 'a+')
                # f.write('%s=%.5f\n' % (file_name, indi.acc))
                

                f = open('./populations/after_%s.json' % (file_name[4:6]), 'r')
                info = json.load(f)

                individual = defaultdict(list)
                individual["file_name"] = file_name
                individual["accuracy"] = indi.acc

                info["cache"].append(individual)
                json_str = json.dumps(info)
                with open('./populations/after_%s.json' % (file_name[4:6]), 'w') as json_file:
                    json_file.write(json_str)
                
                f.flush()
                f.close()
        """
        once the last individual has been pushed into the gpu, the code above will finish.
        so, a while-loop need to be insert here to check whether all GPU are available.
        Only all available are available, we can call "the evaluation for all individuals
        in this generation" has been finished.
        """
        
        if has_evaluated_offspring:
            all_finished = False
            while all_finished is not True:
                # 推迟执行的秒数。
                time.sleep(300)
                # all_finished = GPUTools.all_gpu_available()
                f = open('./populations/after_%s.json' % (file_name[4:6]), 'r')
                info = json.load(f)
                f.flush()
                f.close()

                # TODO: 查看当前个体是否都训练完成
                t_cache = set()
                for i in info["cache"]:
                    t_cache.add(i["file_name"])
                all_finished = True
                for i in self.individuals:
                    if i.id not in t_cache:
                        all_finished = False
                        break
        
        # TODO: 进程似乎不会主动消亡
        # for p in p_l: p.join()

        """
        the reason that using "has_evaluated_offspring" is that:
        If all individuals are evaluated, there is no needed to wait for 300 seconds indicated in line#47
        """
        """
        When the codes run to here, it means all the individuals in this generation have been evaluated, then to save to the list with the key and value
        Before doing so, individuals that have been evaluated in this run should retrieval their fitness first.
        """

        if has_evaluated_offspring:
            file_name = './populations/after_%s.json' % (self.individuals[0].id[4:6])
            # TODO: after_文件是什么时候创建的?
            assert os.path.exists(file_name) == True
            f = open(file_name, 'r')
            
            # fitness_map = {}
            # for line in f:
            #     if len(line.strip()) > 0:
            #         line = line.strip().split('=')
            #         fitness_map[line[0]] = float(line[1])
            # f.close()
            info = json.load(f)
            fitness_map = {}
            for i in info["cache"]:
                fitness_map[i["file_name"]] = float(i["accuracy"])

            for indi in self.individuals:
                if indi.acc == -1:
                    if indi.id not in fitness_map:
                        self.log.warn(
                            'The individuals have been evaluated, but the records are not correct, the fitness of %s does not exist in %s, wait 120 seconds'
                            % (indi.id, file_name))
                        sleep(120)
                    indi.acc = fitness_map[indi.id]
        else:
            self.log.info('None offspring has been evaluated')

        Utils.save_fitness_to_cache(self.individuals)
