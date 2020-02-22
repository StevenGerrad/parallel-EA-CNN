
'''
import multiprocessing
import os

def run_proc(name):
    print('Child process {0} {1} Running '.format(name, os.getpid()))

if __name__ == '__main__':
    print('Parent process {0} is Running'.format(os.getpid()))
    for i in range(5):
        p = multiprocessing.Process(target=run_proc, args=(str(i),))
        print('process start')
        p.start()
    p.join()
    print('Process close')


'''
'''
# from utils import Utils, GPUTools
import importlib
from multiprocessing import Process
import time, os, sys
# sys.path.append("..")

from asyncio.tasks import sleep

from collections import defaultdict, OrderedDict
import json

# import scripts.indi0000

cpu_id = 'test'
file_name = 'indi0000'

# module_name = 'scripts.%s' % (file_name)
# if module_name in sys.modules.keys():
#     # self.log.info('Module:%s has been loaded, delete it' % (module_name))
#     del sys.modules[module_name]
#     # import_module: 根据字符串导入模块
#     _module = importlib.import_module('.', module_name)
# else:
#     _module = importlib.import_module('.', module_name)

# # getattr() 函数用于返回一个对象属性值。
# _class = getattr(_module, 'TrainModel')
# cls_obj = _class()
# p = Process(target=cls_obj.process, args=(
#         '%d' % (cpu_id),
#         file_name,
#     ))
# p.start()


# import importlib


# 绝对导入
a = importlib.import_module("scripts.indi0000")
a.show()
# show A

# 相对导入
b = importlib.import_module(".indi0000", "scripts")
b.show()
# show B

'''

'''
from collections import defaultdict, OrderedDict
import json
import os

if os.path.exists('./test_data.json') != True:
    after_dict = {
        'version': "1.0",
        'cache': [],
        'explain': {
            'used': True,
            'details': "this is for after evaluate",
        }
    }
    json_str = json.dumps(after_dict)
    with open('./test_data.json', 'w') as json_file:
        json_file.write(json_str)

f = open('./test_data.json', 'r')
info = json.load(f)

individual = defaultdict(list)
individual["file_name"] = 0
individual["accuracy"] = 1.0

info["cache"].append(individual)
json_str = json.dumps(info)
with open('./test_data.json', 'w') as json_file:
    json_file.write(json_str)

'''

