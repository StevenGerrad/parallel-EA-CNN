import numpy as np
import hashlib
import copy
import random
import math
import torch

from genetic.component import Edge, Vertex

from collections import defaultdict, OrderedDict
import json

class Individual(object):
    def __init__(self, params, indi_no):
        '''
        传入参数为json(字典)数据
        '''
        self.acc = -1.0  # 若结果非0, 在utils.load_population 中有处理
        self.id = indi_no  # for record the id of current individual
        self.number_id = 0  # for record the latest number of basic unit
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_size_channel = params['num_class']

        # TODO: output_channles为list, 是用来表示啥的???
        self.output_channles = params['output_channel']

        self.learning_rate = params['learning_rate']

        self.params = params
        

    def initialize(self):
        t_vertices = self.params["net"]["vertices"]
        t_edges = self.params["net"]["edges"]

        self.vertices = []
        for ver in t_vertices:
            t_l = Vertex(edges_in=set(),
                         edges_out=set(),
                         type=ver["type"],
                         inputs_mutable=ver["inputs_mutable"],
                         outputs_mutable=ver["outputs_mutable"],
                         properties_mutable=ver["properties_mutable"])
            self.vertices.append(t_l)

        self.edges = []
        for edg in t_edges:
            t_e = Edge(from_vertex=self.vertices[edg["from_vertex"]],
                       to_vertex=self.vertices[edg["to_vertex"]],
                       type=edg["type"])
            self.edges.append(t_e)

        for i, ver in enumerate(self.vertices):
            if "edges_in" in t_vertices[i]:
                for j in t_vertices[i]["edges_in"]:
                    ver.edges_in.add(self.edges[j])
            if "edges_out" in t_vertices[i]:
                for j in t_vertices[i]["edges_out"]:
                    ver.edges_out.add(self.edges[j])
        # self.units = []

    def calculate_flow(self):
        '''
        按顺序计算神经网络每层的输入输出参数
        '''
        # self.vertices[0].input_channel = self.input_size_channel
        self.vertices[0].input_channel = self.image_channel     # 更名为image_channel
        # self.vertices[0].output_channel = self.input_size_channel
        # self.vertices[-1].input_channel = self.output_size_channel
        # self.vertices[-1].output_channel = self.output_size_channel

        for i, vertex in enumerate(self.vertices[1:], start=1):
            vertex.input_channel = 0

            for edge in vertex.edges_in:
                edge.input_channel = edge.from_vertex.input_channel
                edge.output_channel = int(edge.input_channel * edge.depth_factor)
                vertex.input_channel += edge.output_channel

    def __str__(self):
        '''
        由Population._str_调用, 写入种群记录json文件
        '''
        # _str=''
        # for i, vertex in enumerate(self.vertices[1:], start=1):
        #     _str.join('vertex [', i, '].{}'.format(vertex.input_channel))
        #     for edge in vertex.edges_in:
        #         f_ver = self.vertices.index(edge.from_vertex)
        #         if edge.type == 'identity': f_h = 'N'
        #         else: f_h = edge.filter_half_height
        #         if edge.type == 'identity': f_w = 'N'
        #         else: f_w = edge.filter_half_width
        #         _str.join(', {}.{}_s{},{},{}'.format(f_ver, edge.type[0], edge.stride_scale, f_h, f_w))
        #     _str.join('\n')
        # _str.join('[calculate_flow] finish\n')

        net = defaultdict(list)

        for ver in self.vertices:
            t_vertex = defaultdict(list)

            for edg in ver.edges_in:
                t_vertex["edges_in"].append(self.edges.index(edg))
            for edg in ver.edges_out:
                t_vertex["edges_out"].append(self.edges.index(edg))
            t_vertex["type"] = ver.type
            t_vertex["inputs_mutable"] = ver.inputs_mutable
            t_vertex["outputs_mutable"] = ver.outputs_mutable
            t_vertex["properties_mutable"] = ver.properties_mutable

            net["vertices"].append(t_vertex)
        
        for edg in self.edges:
            t_edge = defaultdict(list)

            t_edge["from_vertex"] = self.vertices.index(edg.from_vertex)
            t_edge["to_vertex"] = self.vertices.index(edg.to_vertex)
            t_edge["type"] = edg.type

            net["edges"].append(t_edge)

        # return '\n'.join(_str)
        return json.dumps(net)

    def uuid(self):
        '''
        编辑神经网络结构序列，要保证同一结构网络序列一致
        '''
        _str = []
        # 先对edges进行排序，方便后续处理
        e_list = []
        for edg in self.edges:
            ind_f = self.vertices.index(edg.from_vertex)
            ind_t = self.vertices.index(edg.to_vertex)
            e_list.append((ind_f, ind_t, edg))
        e_list = sorted(e_list, key=lambda x: (x[0], x[1]))
        
        for index, vert in enumerate(self.vertices):
            _sub_str = []
            # 处理vertex层
            if vert.type == 'linear':
                _sub_str.append('linear')
            elif vert.type == 'bn_relu':
                _sub_str.append('bn_relu')
            elif vert.type == 'Global Pooling':
                # Global Pooling 实际上应包含最后的MLP层，但不写也没关系
                _sub_str.append('Global Pooling')
            # 处理edges_in
            for inf_f, _, edg in e_list:
                if edg in vert.edges_in:
                    if edg.type == 'identity':
                        _sub_str.append('identity')
                    elif edg.type == 'conv':
                        _sub_str.append('conv')
                        _sub_str.append('from vertex%d' % (ind_f))
                        _sub_str.append('depth_factor:%d' % (edg.depth_factor))
                        _sub_str.append('filter_half_width:%d' % (edg.filter_half_width))
                        _sub_str.append('filter_half_height:%d' % (edg.filter_half_height))
                        _sub_str.append('stride_scale:%d' % (edg.stride_scale))
            
            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        
        _final_str_ = '-'.join(_str)
        _final_utf8_str_= _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def file_string(self):
        '''
        生成pytorch文件中的
        '''
        # 先理顺卷积网络整体结构
        self.calculate_flow()
        # layer层初始化语句
        unit_list = []
        unit_list.append('self.globalPool = torch.nn.AdaptiveAvgPool2d((1, 1))')
        unit_list.append('self.layer_vertex = torch.nn.ModuleList()')
        for i, vertex in enumerate(self.vertices):
            _str = ''
            if vertex.type == 'bn_relu':
                _str += 'self.layer_vertex.append(torch.nn.Sequential('
                _str +=     'torch.nn.BatchNorm2d(%d),' % (vertex.input_channel)
                _str +=     'torch.nn.ReLU(inplace=True)))'
            elif vertex.type == 'Global Pooling':
                _str += 'self.layer_vertex.append(torch.nn.Sequential('
                _str += 'torch.nn.Linear(%d, %d)))' % (vertex.input_channel, self.output_size_channel)
            else:
                _str += 'self.layer_vertex.append(None)'
            _str += '\n'
            unit_list.append(_str)

        unit_list.append('self.layer_edge = torch.nn.ModuleList()')
        for i, edge in enumerate(self.edges):
            _str = ''
            if edge.type == 'conv':
                _str += 'self.layer_edge.append(torch.nn.Conv2d(%d,' % (edge.input_channel)
                _str +=                        '%d,' % (edge.output_channel)
                _str +=                        'kernel_size=(%d,' % (edge.filter_half_height * 2 + 1)
                _str +=                                     '%d),' % (edge.filter_half_width * 2 + 1)
                _str +=                        'stride=pow(2, %d),' % (edge.stride_scale)
                _str +=                        'padding=(%d, %d)))' % (edge.filter_half_height, edge.filter_half_width)
                # TODO: 暂时未解决权值继承问题
                # if edge.model_id != -1 or self.parent_model == None:
                #     temp.weight = self.parent_model.layer_edge[i].weight
                # self.layer_edge.append(temp)
            else:
                _str += 'self.layer_edge.append(None)'
            _str += '\n'
            unit_list.append(_str)
        
        # 向前传播forward语句
        forward_list = []
        forward_list.append('block_h = input.shape[0]')
        forward_list.append('x0 = input')
        for index, vert in enumerate(self.vertices[1:], start=1):
            # _str = ''
            # 处理edges in的过程
            if len(vert.edges_in) == 1:
                # 若只有一条链接边
                edge = list(vert.edges_in)[0]
                if edge.type == 'conv':
                    ind_edg = self.edges.index(edge)
                    # _str += 'x%d = self.layer_edge[%d](x%d)' % (index, ind_edg, index - 1)
                    forward_list.append('x%d = self.layer_edge[%d](x%d)' % (index, ind_edg, index - 1))
                else:
                    # _str += 'x%d = x%d' % (index, index - 1)
                    forward_list.append('x%d = x%d' % (index, index - 1))
                # _str += '\n'
            else:
                _str = ''
                # 本vertex接收链接数>1
                ind = []
                # 处理链接conv需要提前计算的语句
                for j, edg in enumerate(self.vertices[index].edges_in):
                    ind_edg = self.edges.index(edg)
                    ind_x = self.vertices.index(edg.from_vertex)
                    if edg.type == 'conv':
                        # _str += 'e%d = self.layer_edge[%d](x%d)\n' % (ind_edg, ind_edg, ind_x)
                        forward_list.append('e%d = self.layer_edge[%d](x%d)\n' % (ind_edg, ind_edg, ind_x))
                    else:
                        # _str += 'e%d = x%d\n' % (ind_edg, ind_x)
                        forward_list.append('e%d = x%d\n' % (ind_edg, ind_x))
                    ind.append(ind_edg)
                # 处理矩阵拼接语句
                _str += 'x%d = torch.cat([e%d' % (index, ind[0])
                for i in ind[1:]:
                    _str += ',e%d' % (i)
                _str += '], dim=1)'
                forward_list.append(_str)

            # 处理vertex的计算
            if self.vertices[index].type == 'linear':
                # forward_list.append(_str)
                continue
            elif self.vertices[index].type == 'bn_relu':
                # _str += 'x%d = self.layer_vertex[%d](x%d)\n' % (index, index, index)
                forward_list.append('x%d = self.layer_vertex[%d](x%d)\n' % (index, index, index))
            elif self.vertices[index].type == 'Global Pooling':
                # _str += 'x%d = self.globalPool(x%d)\n' % (index, index)
                forward_list.append('x%d = self.globalPool(x%d)\n' % (index, index))
                # _str += 'x%d = torch.squeeze(x%d, 3)\n' % (index, index)
                forward_list.append('x%d = torch.squeeze(x%d, 3)\n' % (index, index))
                # _str += 'x%d = torch.squeeze(x%d, 2)\n' % (index, index)
                forward_list.append('x%d = torch.squeeze(x%d, 2)\n' % (index, index))
                # _str += 'x%d = self.layer_vertex[%d](x%d)\n' % (index, index, index)
                forward_list.append('x%d = self.layer_vertex[%d](x%d)\n' % (index, index, index))
            # forward_list.append(_str)
        
        forward_list.append('return x%d' % (len(self.vertices)-1))
        return unit_list, forward_list

    def add_edge(self,
                 from_vertex_id,
                 to_vertex_id,
                 edge_type='identity',
                 depth_factor=1,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=0):
        """
        Adds an edge to the DNA graph, ensuring internal consistency.
        """
        edge = Edge(from_vertex=self.vertices[from_vertex_id],
                    to_vertex=self.vertices[to_vertex_id],
                    type=edge_type,
                    depth_factor=depth_factor,
                    filter_half_width=filter_half_width,
                    filter_half_height=filter_half_height,
                    stride_scale=stride_scale)
        edge.model_id = -1

        self.edges.append(edge)
        self.vertices[from_vertex_id].edges_out.add(edge)
        self.vertices[to_vertex_id].edges_in.add(edge)

        return edge

    def mutate_layer_size(self, v_list=[], s_list=[]):
        for i in range(len(v_list)):
            self.vertices[v_list[i]].outputs_mutable = s_list[i]

    def add_vertex(self, after_vertex_id, vertex_type='linear', edge_type='identity'):
        '''
        3.0: 所有 vertex 和 edg 中记录的都是引用
        '''
        changed_edge = None
        # 先寻找那条应该被移除的边, 将其删除
        for i in self.vertices[after_vertex_id - 1].edges_out:
            if i.to_vertex == self.vertices[after_vertex_id]:
                self.vertices[after_vertex_id - 1].edges_out.remove(i)
                break
        for i in self.vertices[after_vertex_id].edges_in:
            if i.from_vertex == self.vertices[after_vertex_id - 1]:
                self.vertices[after_vertex_id].edges_in.remove(i)
                break
        for i, edge in enumerate(self.edges):
            if edge.from_vertex == self.vertices[
                    after_vertex_id - 1] and edge.to_vertex == self.vertices[after_vertex_id]:
                changed_edge = self.edges[i]

        # 创建新的 vertex, 并加入队列
        vertex_add = Vertex(edges_in=set(), edges_out=set(), type=vertex_type)
        self.vertices.insert(after_vertex_id, vertex_add)

        # 创建新的 edge, 并加入队列
        if edge_type == 'conv':
            depth_f = max(1.0, random.random() * 4)
            filter_h = 1
            filter_w = 1
            stride_s = math.floor(random.random() * 2)
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='conv',
                             depth_factor=depth_f,
                             filter_half_height=filter_h,
                             filter_half_width=filter_w,
                             stride_scale=0)
            edge_add1.model_id = -1
        else:
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='identity')
            edge_add1.model_id = -1
        # 取代的那条边后移
        changed_edge.from_vertex = self.vertices[after_vertex_id]
        # edge_add2 = Edge(from_vertex=self.vertices[after_vertex_id],to_vertex=self.vertices[after_vertex_id + 1])
        self.edges.append(edge_add1)
        # self.edges.append(edge_add2)

        self.vertices[after_vertex_id - 1].edges_out.add(edge_add1)
        vertex_add.edges_in.add(edge_add1), vertex_add.edges_out.add(changed_edge)
        self.vertices[after_vertex_id + 1].edges_in.add(changed_edge)

    def has_edge(self, from_vertex_id, to_vertex_id):
        vertex_before = self.vertices[from_vertex_id]
        vertex_after = self.vertices[to_vertex_id]
        for edg in self.edges:
            if edg.from_vertex == vertex_before and edg.to_vertex == vertex_after:
                return True
        return False


######################################################################################################
#
# 
# 
# 
# 
######################################################################################################


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for i in range(self.pop_size):
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            self.number_id += 1
            param_item = self.params
            param_item["net"] = self.params["populations"][i]
            indi = Individual(param_item, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            # indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        '''
        由EvolveCNN.initialize_population和.environment_selection调用，将种群写入种群记录文件
        '''
        # _str = []
        # for ind in self.individuals:
        #     _str.append(str(ind))
        #     _str.append('-' * 100)
        # return '\n'.join(_str)

        _str = defaultdict(list)
        for ind in self.individuals:
            ind_item = json.loads(str(ind))
            _str["populations"].append(ind_item)
        return json.dumps(_str)



def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())


def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)


if __name__ == '__main__':
    test_individual()
    #test_population()
