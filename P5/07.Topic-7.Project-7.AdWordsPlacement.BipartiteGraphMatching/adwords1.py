import sys
import pandas as pd
import numpy as np
import time
import pickle

def getArguments():
    if len(sys.argv) != 2:
        print "adwords.py <algo_name>"
        sys.exit(0)
    return sys.argv[1]

def getData():
    ds = pd.read_csv('./bidder_dataset.csv')

    queries = pd.read_csv('./queries.txt', header=None, sep = '\n')
    queries1 = np.loadtxt('./queries.txt', dtype = 'string', delimiter='\n')
    #print queries1
    return ds,queries1

def getHighestBidder(neighbors, advertiser_wise_budget, bankrupted):
    #Remove any bidder whose budget is over.
    neighbors = neighbors[neighbors['Advertiser'] != bankrupted]
    if neighbors.empty:
        #print neighbors
        return neighbors
    #Highest Bidder
    highest_bidder = neighbors.loc[neighbors['Bid Value'].idxmax()]
    #print highest_bidder
    if advertiser_wise_budget[highest_bidder['Advertiser']]-highest_bidder['Bid Value'] <= 0:
        bankrupted = highest_bidder['Advertiser']
        highest_bidder = getHighestBidder(neighbors, advertiser_wise_budget, bankrupted)
    return highest_bidder



def applyAlgorithm(algo_name, dataset, queries):
    # print(dataset['Keyword'])
    # print(queries[0][0])

    # print np.shape(queries)
    """
    init =  time.time()
    query_wise_neighbors = {}
    for r in queries:
        neighbors = dataset.loc[dataset['Keyword'] == r]
        if r not in query_wise_neighbors:
            query_wise_neighbors[r] = neighbors
    finish =  time.time()
    print finish - init
    with open('neighbors.pickle', 'wb') as handle:
        pickle.dump(query_wise_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    with open('neighbors.pickle', 'rb') as handle:
        query_wise_neighbors = pickle.load(handle)

    advertiser_wise_budget = {}
    for row in dataset.iterrows():
        if np.isnan(row[1]['Budget']) == False:
            advertiser_wise_budget[row[1]['Advertiser']] = row[1]['Budget']

    #print advertiser_wise_budget

    # c = 0
    # for k,v in query_wise_neighbors.iteritems():
    #     c+=1
    #     print c
    #     print k
    #     print v


    if algo_name.lower() == 'balance':
        pass



    elif algo_name.lower() == 'greedy':
        c = 0
        revenue = 0
        for query in queries:
            c = c+1
            #print c
            neighbors = query_wise_neighbors[query]
            sum = 0
            #print query
            #print neighbors
            for neighbor in neighbors['Advertiser']:
                #print advertiser_wise_budget[neighbor]
                sum += advertiser_wise_budget[neighbor]
            #print sum
            if sum == 0:
                continue
            else:
                bankrupted = None
                hBidder = getHighestBidder(neighbors, advertiser_wise_budget, bankrupted)
                if hBidder.empty:
                    continue
                #print hBidder
                #print advertiser_wise_budget[hBidder['Advertiser']]
                advertiser_wise_budget[hBidder['Advertiser']] = advertiser_wise_budget[hBidder['Advertiser']] - hBidder['Bid Value']
                #print advertiser_wise_budget[hBidder['Advertiser']]
                revenue += hBidder['Bid Value']

            #print "======================================="

        print revenue




    elif algo_name.lower() == 'msvv':
        pass

if __name__ == "__main__":
    algo_name = getArguments()
    print "Using the",algo_name,"approach"
    (dataset, queries) = getData()

    applyAlgorithm(algo_name, dataset, queries)
