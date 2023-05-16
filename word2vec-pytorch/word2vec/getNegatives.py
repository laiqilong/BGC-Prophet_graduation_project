import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing
import sys
import torch

def throw_error(e):
    raise e

def getN(intervalContexts, intervalCenters, i, sampleWeights, k, n, min_count):
    print(i, 'subprocess!')
    # lenCenters = len(centers)
    # centers = None
    print("get contexts subset", i)
    generator = RandomGenerator(samplingWeights=sampleWeights)
    negatives = []
    print(f"Length of this contexts: {len(intervalContexts)}")
    for j in tqdm(range(len(intervalContexts)), desc=f'Get negatives subprocess {i}', leave=True):
    # for context in tqdm(contexts, desc=f'Get negatives subprocess {i}', leave=True):
        context = intervalContexts[j]
        center = intervalCenters[j]
        negative = []
        while(len(negative)<len(context)*k):
            neg = generator.draw()
            if (neg not in context) and (neg != center):
                negative.append(neg)
        negatives.append(negative)
    with open(f'./Negatives/output_negatives_{i}.pkl', 'wb')as fp:
        pickle.dump(negatives, fp)
    # return negatives, i

class RandomGenerator:

    def __init__(self, samplingWeights) -> None:
        self.population = list(range(0, len(samplingWeights)))
        self.samplingWeights = samplingWeights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = np.random.choice(self.population, int(1e8), p=self.samplingWeights)
            self.i = 0
        self.i+=1
        return self.candidates[self.i-1]

class getNegatives:
    def __init__(self, centersPath, contextsPath, min_count, i) -> None:
        self.centersPath = centersPath
        self.contextsPath = contextsPath
        self.min_count = min_count
        self.k = 5
        self.i = i
        self.negatives = []
        self.__getNegatives()
        # self.__saveNegatives()

    def __getNegatives(self):
        with open(f'output_word_frequency_{self.min_count}.pkl', 'rb') as fp:
            self.word_frequency = pickle.load(fp)

        with open(f'output_centers_{self.min_count}.pkl', 'rb') as fp:
            self.centers = pickle.load(fp)
        with open(f'output_contexts_{self.min_count}.pkl', 'rb') as fp:
            self.contexts = pickle.load(fp)
        sampleWeights = np.array(list(self.word_frequency.values())) ** 0.75
        sampleWeights /= sampleWeights.sum()
        contextsLen = len(self.contexts)
        centersLen = len(self.centers)
        def miniPool(n, istart, iend):
            pool = multiprocessing.Pool(processes=40, maxtasksperchild=1)
            for i in range(istart, iend):
                intervalContexts = self.contexts[int(i*contextsLen/n):int((i+1)*contextsLen/n)]
                intervalCenters = self.centers[int(i*centersLen/n):int((i+1)*centersLen/n)]
                pool.apply_async(getN, args=(intervalContexts, intervalCenters, i, sampleWeights, self.k, n, self.min_count), error_callback=throw_error)
            pool.close()
            pool.join()
            # for job in jobs:
            #     results.append(job.get())
            # results.sort(key=lambda a:a[1])
            # for res in results:
            #     self.negatives += res[0]
            # pool.terminate()
        if self.i == '1':
            miniPool(120, 0, 20)
        elif self.i == '2':
            miniPool(120, 20, 40)
        elif self.i == '3':
            miniPool(120, 40, 60)
        elif self.i == '4':
            miniPool(120, 60, 80)
        elif self.i == '5':
            miniPool(120, 80, 100)
        else:
            miniPool(120, 100, 120)
        # if self.i == '1':
        #     miniPool(80, 0, 40)
        # else:
        #     miniPool(80, 40, 80)

    # def __saveNegatives(self):
    #     with open(f'output_negatives_{self.min_count}_{self.i}.pkl', 'wb') as fp:
    #         pickle.dump(self.negatives, fp)

if __name__ == '__main__':
    i = sys.argv[1]
    getNegatives('output_centers_5.pkl','output_contexts_5.pkl', min_count=5, i=i)