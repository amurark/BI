#!/usr/local/bin
import sys
import csv
import pprint
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

pp = pprint.PrettyPrinter(indent = 1)
def readAttrList():
    with open('./data/fb_caltech_small_attrlist.csv', 'rb') as csvfile:
        attrlist = csv.reader(csvfile, delimiter = ',')
        header = []
        data = []
        for i,row in enumerate(attrlist):
            if i == 0:
                header = row
            else:
                data.append(row)
    return header,data

def reduceFunc(x,y):
    if int(y[0]) not in x:
        x[int(y[0])] = []
    if int(y[1]) not in x:
        x[int(y[1])] = []
    x[int(y[0])].append(int(y[1]))
    x[int(y[1])].append(int(y[0]))
    return x

def readEdgeList():
    with open('./data/fb_caltech_small_edgelist.txt', 'rb') as txtfile:
        edgelist = txtfile.readlines()
        somelist = map((lambda x: x.strip().split(" ")),edgelist)
        dictionary = reduce(reduceFunc, somelist, {})
    return dictionary

#Move i from i to j
def compute_composite_modularity_gain(i,j,V,E,communities,adj_list,attr):
    #print i,j,alpha
    m = len(E)
    links = set(communities[j]).intersection(adj_list[i])
    sum_deg = 0
    simA = 0
    for node in communities[j]:
        sum_deg += len(adj_list[node])
        result = cosine_similarity(attr[i], attr[node])
        #result = 1 - spatial.distance.cosine(attr[i], attr[node])
        simA += float(result[0][0])
    delq_newman = (1/(float(2*m))) * (len(links) - float(len(adj_list[i])/(float(2*m))) * float(sum_deg))
    delq_attr = simA/float(len(communities[j]))
    delq = (alpha * delq_newman) + ((1-alpha) * delq_attr)
    return delq


def phase1(attr, adj_list):
    communities = {}
    V = range(0,324)
    #initialize each node to a community
    for node in V:
        communities[node] = [node]
    #Create edge list
    E = []
    for node in adj_list.keys():
        for i in adj_list[node]:
            E.append(str(node)+"_"+str(i))

    cnt= 0
    while(cnt< 1):
        for i in V:
            max_gain = 0
            max_index = 0
            for j in V:
                if i != j:
                    comp_mod_gain = compute_composite_modularity_gain(i,j,V,E,communities,adj_list,attr)
                    if comp_mod_gain > max_gain:
                        max_gain = comp_mod_gain
                        max_index = j
            print max_gain, max_index
            if max_gain > 0:
                communities[max_index].append(i)
                for ele in communities[max_index]:
                    communities[ele] = communities[max_index]


        cnt+= 1
    pp.pprint(communities)

if __name__ == '__main__':
    global alpha
    arg_vec = sys.argv
    if len(arg_vec) <= 1:
        sys.exit("Insufficient variables: python sac1.py <alpha values>")
    alpha_val = arg_vec[1]
    if float(alpha_val) not in [0, 0.5, 1]:
        sys.exit("Invalid alpha value(0, 0.5, 1)")
    alpha = float(alpha_val)
    header,attrs = readAttrList()
    #print header
    #print(attrs)
    adj_list = readEdgeList()
    #pp.pprint(adj_list)
    phase1(attrs, adj_list)
