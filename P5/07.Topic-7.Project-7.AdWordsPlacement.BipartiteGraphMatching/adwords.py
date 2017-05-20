import sys
import pandas as pd
import numpy as np
import time
import pickle
import math

#Get the argument for the type of algorith. Gracefully exit if there is no argument or the argument is wrong.
def getArguments():
    if len(sys.argv) != 2:
        sys.exit("adwords.py <algo_name>")
    elif sys.argv[1].lower() not in ['mssv','greedy','balance']:
        sys.exit("Invalid argument, allowed:(mssv, greedy, balance)")
    return sys.argv[1]

#Read data out of the CSV file and queries from the text file.
def getData():
    ds = pd.read_csv('./bidder_dataset.csv')

    queries = pd.read_csv('./queries.txt', header=None, sep = '\n')
    queries1 = np.loadtxt('./queries.txt', dtype = 'string', delimiter='\n')
    return ds,queries1


#Apply the required algorithm
def applyAlgorithm(algo_name, dataset, queries):

    #Create a query-wise-neighbors dictionary.
    query_wise_neighbors = {}
    for r in queries:
        neighbors = dataset.loc[dataset['Keyword'] == r]
        if r not in query_wise_neighbors:
            query_wise_neighbors[r] = neighbors
    """
    with open('neighbors.pickle', 'wb') as handle:
        pickle.dump(query_wise_neighbors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('neighbors.pickle', 'rb') as handle:
        query_wise_neighbors = pickle.load(handle)
    """

    #Create a advertiser-wise-budget dictionary.
    advertiser_wise_budget = {}
    for row in dataset.iterrows():
        if np.isnan(row[1]['Budget']) == False:
            advertiser_wise_budget[row[1]['Advertiser']] = row[1]['Budget']



    total_revenue = 0
    #Iterate 100 times for shuffled permutations of the query sequence, and get results.
    for i in range(0,100):
        #Give a seed value of 0.
        np.random.seed(0)
        #Shuffle queries
        np.random.shuffle(queries)
        #To hold "per iteration" revenue
        revenue = 0
        #To hold optimal revenue
        opt_revenue = 0
        graph = {}
        #This dictionary stores the updated budget for each advertiser after each successful bid.
        advertiser_wise_budget_copy = dict(advertiser_wise_budget)
        for query in queries:
            sum = 0
            if query not in graph:
                #Store all advertisers who bid for this query
                neighbors = query_wise_neighbors[query]
                graph[query] = {}
                #For all advertisers for this query, create a dictionary
                for neighbor in neighbors.iterrows():
                    adv = neighbor[1]['Advertiser']
                    bid_val = neighbor[1]['Bid Value']
                    graph[query][adv] = bid_val
            #To check if all neighbors have spent their budget.
            for neighbor in graph[query].keys():
                sum += advertiser_wise_budget_copy[neighbor]
                bid_val = graph[query][neighbor]
                #If a particular neighbor has a remaining budget less than its bid value, ignore it for the bidding.
                if advertiser_wise_budget_copy[neighbor]-bid_val < 0:
                    del graph[query][neighbor]
            #If selected algo is 'balance'
            if algo_name.lower() == 'balance':
                if sum == 0 or not graph[query]:
                    continue
                else:
                    #Get the highest bidder for the QUERY and update its budget.
                    maximum = 0
                    highest_bidder = 0
                    for key in graph[query].keys():
                        if advertiser_wise_budget_copy[key] > maximum:
                            maximum = advertiser_wise_budget_copy[key]
                            highest_bidder = key
                    advertiser_wise_budget_copy[highest_bidder] = float(advertiser_wise_budget_copy[highest_bidder]) - float(graph[query][highest_bidder])
                    revenue += graph[query][highest_bidder]
            #If selected algo is 'greedy'
            elif algo_name.lower() == 'greedy':
                if sum == 0 or not graph[query]:
                    continue
                else:
                    #Get the highest bidder for the QUERY and update its budget.
                    highest_bidder = max(graph[query], key=lambda key: graph[query][key])
                    advertiser_wise_budget_copy[highest_bidder] = float(advertiser_wise_budget_copy[highest_bidder]) - float(graph[query][highest_bidder])
                    revenue += graph[query][highest_bidder]
            #If selected algo is 'mssv'
            elif algo_name.lower() == 'mssv':
                if sum == 0 or not graph[query]:
                    continue
                else:
                    #Get the highest bidder for the QUERY and update its budget.
                    # Xu
                    xu = {key : (advertiser_wise_budget[key] - advertiser_wise_budget_copy.get(key,0))/advertiser_wise_budget[key] for key in graph[query].keys()}
                    #psi(Xu)
                    psi_xu = {key : 1 - math.exp(xu[key]-1) for key in xu.keys()}
                    highest_bidder = max(psi_xu, key=lambda key: psi_xu[key]*graph[query][key])
                    advertiser_wise_budget_copy[highest_bidder] = float(advertiser_wise_budget_copy[highest_bidder]) - float(graph[query][highest_bidder])
                    revenue += graph[query][highest_bidder]
            #Calculate optimal matching OPT
            n = query_wise_neighbors[query]
            hb = list(n['Bid Value'])
            opt_revenue += max(hb)
        total_revenue += revenue
    total_revenue /= 100
    ans = total_revenue/opt_revenue
    print "Total Revenue: %.2f"%total_revenue
    print "Competitve Ratio: %.2f"%ans


if __name__ == "__main__":
    algo_name = getArguments()
    print "Using the",algo_name,"approach"
    (dataset, queries) = getData()

    applyAlgorithm(algo_name, dataset, queries)
