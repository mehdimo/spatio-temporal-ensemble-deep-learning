from pybloom import BloomFilter, ScalableBloomFilter
import numpy as np
import sys


class TemporalBloomFilter:
    '''
    Creates several bloom filters

    :param BFsNo: number of level of depth in history
    :param deltas: list of discretization factors
    :param modelsNo: number of individual models
    A total of BFsNo*len(deltas) bloom filter is created.
    '''

    def __init__(self, BFsNo, deltas, modelsNo=3):
        '''create bloom filters'''
        BFs = []
        self.deltas = deltas
        self.modelsNo = modelsNo
        self.BFsNo = BFsNo
        deltas_no = len(deltas)
        for i in range(deltas_no * BFsNo):
            bf = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)
            BFs.append(bf)
        self.BFs = np.array(BFs)
        self.BFs = self.BFs.reshape(deltas_no, BFsNo)

    def __intv(self, a, delta):
        '''
        Convert a value 'a' to its discretized counterpart.
        :param a: input value
        :param delta: discretization factor
        :return: discretized value as a string
        '''
        return str(int(a // delta) * delta)

    def __intva(self, ar, delta):
        '''
        Return a string of discretized items in an array
        :param ar: list of float values
        :param delta: discritization factor
        :return: string of discretized values seperated by comma
        '''
        st = ''
        for a in ar:
            st += self.__intv(a, delta) + ','
            st = st[:-1]
        return st

    def fill(self, data):
        '''
        Fill all bloom filters based of the discretized values
        :param data: list of values
        :return None
        '''
        for j in range(len(self.deltas)):
            # fill bloom filter
            endt = int(len(data))
            delta = self.deltas[j]
            blmNo = self.BFsNo
            for i in range(blmNo, endt):
                p = []
                for k in range(blmNo + 1):
                    dataPoint = data[i - k]
                    p.append(dataPoint)

                for k in range(blmNo):
                    val = self.__intva(p[:k + 2], delta)
                    self.BFs[j, k].add(val)

    def check(self, df, i):
        '''
        Check if a specific pandas DataFrame item at index i is available in the bloom filters
        :param df: DataFrame containing the float values
        :param i: index of desired item to check
        :return: Array of bloom filters hitted, array of pattern of hitted bloom filters, is there any hit on any bloom filter or not.
        '''
        hit_bloom = 0
        models_Bf_hit = np.array([0.0] * self.modelsNo)
        models_hit_pattern = np.zeros(self.modelsNo)
        # check history of items
        for j in range(len(self.deltas)):
            delta = self.deltas[j]
            for k in range(self.modelsNo):
                colName = 'predict_' + str(k + 1)
                points_q = []
                for q in range(self.BFsNo + 1):
                    t_q = df.at[i - q, colName]
                    points_q.append(t_q)

            for r in range(self.BFsNo - 1, -1, -1):
                val = self.__intva(points_q[:r + 2], delta)
                if models_hit_pattern[k] == 0:
                    if val in self.BFs[j, r]:
                        models_Bf_hit[k] = (r + 2) / float(delta)
                        models_hit_pattern[k] = r + 2
                        hit_bloom = 1

        return models_Bf_hit, models_hit_pattern, hit_bloom

    def get_size(self):
        '''
        Get the total zize of bloom filters
        :return: Size of bloom filters
        '''
        size = 0
        if len(self.BFs) > 0:
            for i in range(len(self.deltas)):
                for j in range(self.BFsNo):
                    size += sys.getsizeof(self.BFs[i, j])
        return size
