
"""
The set contains the following mutations:

√   • 学习率 ALTER-LEARNING-RATE (sampling details below).
    • 线性 IDENTITY (effectively means “keep training”).
    • 权重 RESET-WEIGHTS (sampled as in He et al. (2015), for example).
√   • 增 INSERT-CONVOLUTION (inserts a convolution at a random location in the “convolutional backbone”, 
        as in Figure 1. The inserted convolution has 3 × 3 filters, strides of 1 or 2 at random, number 
        of channels same as input.May apply batch-normalization and ReLU activation or none at random).
    • 删 REMOVE-CONVOLUTION.
    • 步长 ALTER-STRIDE (only powers of 2 are allowed).
    • 通道数 ALTER-NUMBER-OF-CHANNELS (of random conv.).
    • 滤波器大小 FILTER-SIZE (horizontal or vertical at random, on random convolution, odd values only).
    • INSERT-ONE-TO-ONE (inserts a one-to-one/identity connection, analogous to insert-convolution mutation).
√   • 增跳层 ADD-SKIP (identity between random layers).
    • 删跳层 REMOVE-SKIP (removes random skip).

firstly, three basic operations:add, remove, alter
secondly, the particular operation is chosen based on a probability
"""
import random
import numpy as np
import copy
from utils import StatusUpdateTool, Utils

from collections import defaultdict, OrderedDict
import json

class CrossoverAndMutation(object):
    '''
    can mutate: hidden size, add edge, learning rate, add vertex, 
    '''
    def __init__(self, _log, individuals, _params=None):
        '''
        TODO: 完善log日志语句
        _params: gene_no
        '''
        self.individuals = individuals
        self.params = _params # storing other parameters if needed, such as the index for SXB and polynomial mutation
        self.log = _log
        self.offspring = []

    def process(self):
        '''
        可能出现'没有任何变异'的情况，不能让其发生
        1. 添加边时：添加identity, 则矩阵拼接时需要维度匹配 / 添加conv则需要是设置好参数
        '''
        offspring = []
        for dna in self.individuals:
            mutated_dna = copy.deepcopy(dna)
            mutated_cnt = 0
            while mutated_cnt == 0:
                # .Try the candidates in random order until one has the right connectivity.(Add)
                for from_vertex_id, to_vertex_id in self._vertex_pair_candidates(dna):
                    # 防止每次变异次数过多
                    if random.random() < pow(0.4, mutated_cnt + 1):
                        mutated_cnt += 1
                        self._mutate_structure(mutated_dna, from_vertex_id, to_vertex_id)

                # .Try to mutate learning Rate
                self.mutate_learningRate(mutated_dna)

                # .Mutate the vertex (Add)
                if random.random() > 0.6:
                    mutated_cnt += 1
                    self.mutate_vertex(mutated_dna)

                # .REMOVE-CONVOLUTION TODO:(这个原意是只移除conv类型的edge吗？)
                if random.random() > 0.6:
                    mutated_cnt += self.remove_edge(mutated_dna)
            offspring.append(mutated_dna)

        # 记录编译后个体
        self.offspring = offspring
        # Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])

        # 更新新一代种群个体的id号，并重置accuracy
        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d'%(self.params['gen_no'], i)
            indi.id = indi_no
            indi.acc = -1.0

        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])

        return offspring
    
    def individuals_to_string(self):
        '''
        将individuals存入mutation种群记录文件
        '''
        # _str = []
        # for ind in self.offspring:
        #     _str.append(str(ind))
        #     _str.append('-'*100)
        # return '\n'.join(_str)

        _str = defaultdict(list)
        for ind in self.offspring:
            ind_item = json.loads(str(ind))
            _str["populations"].append(ind_item)
        return json.dumps(_str)

    def _vertex_pair_candidates(self, dna):
        """Yields connectable vertex pairs."""
        from_vertex_ids = self._find_allowed_vertices(dna)
        # if not from_vertex_ids: raise exceptions.MutationException(), 打乱次序
        random.shuffle(from_vertex_ids)

        to_vertex_ids = self._find_allowed_vertices(dna)
        # if not to_vertex_ids: raise exceptions.MutationException()
        random.shuffle(to_vertex_ids)

        for to_vertex_id in to_vertex_ids:
            # Avoid back-connections. TODO: 此处可能会涉及到 拓扑图的顺序判断
            # disallowed_from_vertex_ids, _ = topology.propagated_set(to_vertex_id)
            disallowed_from_vertex_ids = self._find_disallowed_from_vertices(dna, to_vertex_id)
            for from_vertex_id in from_vertex_ids:
                if from_vertex_id in disallowed_from_vertex_ids:
                    continue
                # This pair does not generate a cycle, so we yield it.
                yield from_vertex_id, to_vertex_id

    def _find_allowed_vertices(self, dna):
        ''' 
        TODO: 除第一层(假节点)外的所有vertex_id 
        '''
        return list(range(0, len(dna.vertices)))

    def _find_disallowed_from_vertices(self, dna, to_vertex_id):
        ''' 寻找不可作为起始层索引的：反向链接的，重复连接的Edge '''
        res = list(range(to_vertex_id, len(dna.vertices)))
        # 排查每个 vertex 是否不符合, 即索引在前面的 vertex 的所有 edges_out
        for i, vertex in enumerate(dna.vertices[:to_vertex_id]):
            for edge in vertex.edges_out:
                if dna.vertices.index(edge.to_vertex) == to_vertex_id:
                    if i not in res:
                        res.append(i)
                        continue
        return res

    def _mutate_structure(self, dna, from_vertex_id, to_vertex_id):
        '''
        Adds the edge to the DNA instance.
        '''
        if dna.has_edge(from_vertex_id, to_vertex_id):
            return False
        else:
            # TODO: edge 有两个类型，identity 和 conv (主要调节 stride, 在默认padding补全的情况下)
            # 1. 若数据维度不变，可以用identity， 则需要检查 stride 是否不变
            res = True
            bin_stride = 0
            for vertex_id, vert in enumerate(dna.vertices[from_vertex_id + 1:to_vertex_id],
                                             start=from_vertex_id + 1):
                edg_direct = None
                for edg in vert.edges_in:
                    if edg.from_vertex == dna.vertices[
                            vertex_id - 1] and edg.to_vertex == dna.vertices[vertex_id]:
                        edg_direct = edg
                        break
                if edg_direct.stride_scale != 0:
                    res = False
                    bin_stride += edg_direct.stride_scale
            if res and random.random() > 0.6:
                # print("[add_edge]->identity:", from_vertex_id, to_vertex_id)
                new_edge = dna.add_edge(from_vertex_id, to_vertex_id)
                self.log.info('#%s: Add edge of identity from vertex #%d to #%d' % (dna.id, from_vertex_id, to_vertex_id))
                return res
            # 2. 若数据维度改变(变小)，要用conv
            # print("[add_edge]->conv:", from_vertex_id, to_vertex_id)
            depth_f = max(1.0, random.random() * 4)
            filter_h = 1
            filter_w = 1
            new_edge = dna.add_edge(from_vertex_id,
                                    to_vertex_id,
                                    edge_type='conv',
                                    depth_factor=depth_f,
                                    filter_half_height=filter_h,
                                    filter_half_width=filter_w,
                                    stride_scale=bin_stride)
            self.log.info('#%s: Add edge of identity from vertex #%d to #%d' % (dna.id, from_vertex_id, to_vertex_id))
            return True

    def mutate_learningRate(self, dna):
        '''
        修改学习率，获取一个为随机数的乘数factor
        '''
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        factor = 2 ** random.uniform(-1.0, 1.0)
        mutated_dna.learning_rate = dna.learning_rate * factor

        self.log.info('#%s: Alert the learning rate from %.3f to %.3f' % (dna.id, dna.learning_rate , factor))
        return mutated_dna

    def mutate_vertex(self, dna):
        '''
        添加vertex层
        '''
        # mutated_dna = copy.deepcopy(dna)
        mutated_dna = dna
        # 随机选择一个 vertex_id 插入 vertex
        after_vertex_id = random.choice(self._find_allowed_vertices(dna))
        if after_vertex_id == 0:
            return mutated_dna

        # print('outputs_mutable', dna.vertices[after_vertex_id].outputs_mutable, dna.vertices[after_vertex_id - 1].outputs_mutable)

        # TODO: how it supposed to mutate
        vertex_type = 'linear'
        if random.random() > 0.2:
            vertex_type = 'bn_relu'

        edge_type = 'identity'
        if random.random() > 0.2:
            edge_type = 'conv'

        mutated_dna.add_vertex(after_vertex_id, vertex_type, edge_type)
        
        self.log.info('#%s: Add Vertex after vertex#%d as %s with %s edge' % (dna.id, after_vertex_id, vertex_type, edge_type))
        return mutated_dna

    def remove_edge(self, dna):
        '''
        移除edge(conv / identity)
        dna结构较为简单时可能无法移除
        '''
        mutated_dna = dna
        
        e_list = []
        for edg in mutated_dna.edges:
            ind_f = mutated_dna.vertices.index(edg.from_vertex)
            ind_t = mutated_dna.vertices.index(edg.to_vertex)
            if ind_f == ind_t - 1:
                continue
            else:
                e_list.append(edg)
            
        if len(e_list) == 0:
            return 0
        
        chosed_edge = random.choice(e_list)
        
        ind_f = mutated_dna.vertices.index(chosed_edge.from_vertex)
        ind_t = mutated_dna.vertices.index(chosed_edge.to_vertex)
        for edg in mutated_dna.vertices[ind_f].edges_out:
            if edg.to_vertex == mutated_dna.vertices[ind_t]:
                mutated_dna.vertices[ind_f].edges_out.remove(edg)
                break
        for edg in mutated_dna.vertices[ind_t].edges_in:
            if edg.from_vertex == mutated_dna.vertices[ind_f]:
                mutated_dna.vertices[ind_t].edges_in.remove(edg)
                break
        mutated_dna.edges.remove(chosed_edge)
        
        self.log.info('#%s: Remove edge of %s which is from vertex#%d to #%d' % (chosed_edge.type, dna.id, ind_f, ind_t))
        del chosed_edge
        return 1
        


if __name__ == '__main__':
    ...
