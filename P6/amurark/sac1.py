#!/usr/local/bin
import sys
import csv
import pprint
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from operator import add, div

pp = pprint.PrettyPrinter(indent = 1)
p2_p1_map = {}
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
def compute_composite_modularity_gain(i,j,attrs, adj_list, V, E, community_nodemap, node_communitymap):
    #print i,j,alpha
    m = len(E)/2
    curr_community = community_nodemap[node_communitymap[j]]


    links = set(curr_community).intersection(adj_list[i])
    sum_deg = 0
    simA = 0
    for node in curr_community:
        sum_deg += len(adj_list[node])
        result = cosine_similarity(attrs[i], attrs[node])
        #result = 1 - spatial.distance.cosine(attr[i], attr[node])
        simA += float(result[0][0])
    delq_newman = (1/(float(2*m))) * (len(links) - float(len(adj_list[i])/(float(2*m))) * float(sum_deg))
    delq_attr = simA/float(len(curr_community))
    delq = (alpha * delq_newman) + ((1-alpha) * delq_attr)
    return delq


def phase1(attrs, adj_list, V, E, community_nodemap, node_communitymap, sim_metric, iter_count):
    cnt= 0
    changeFlag = True
    try:
        while(cnt< 15 and changeFlag):
            changeFlag = False
            for i in V:
                max_gain = 0
                max_index = 0
                for j in V:
                    if (i != j) and (node_communitymap[i] != node_communitymap[j]):#modified
                        #comp_mod_gain = compute_composite_modularity_gain(i,j,attrs, adj_list, V, E, community_nodemap, node_communitymap)

                        m = len(E)/2
                        curr_community = community_nodemap[node_communitymap[j]]


                        links = curr_community.intersection(adj_list[i])
                        sum_deg = 0
                        simA = 0
                        for node in curr_community:
                            sum_deg += len(adj_list[node])
                            k1 = str(i)+"_"+str(node)
                            k2 = str(node)+"_"+str(i)
                            if (k1 in sim_metric):#Modified
                                result = sim_metric[k1]
                            elif (k2 in sim_metric):
                                result = sim_metric[k2]
                            else:
                                result = cosine_similarity(attrs[i], attrs[node])
                                sim_metric[k1] = result#Update similarity metrics
                            simA += float(result[0][0])
                        delq_newman = (1/(float(2*m))) * (len(links) - float(len(adj_list[i])/(float(2*m))) * float(sum_deg))
                        delq_attr = simA/float(len(curr_community))
                        comp_mod_gain = (alpha * delq_newman) + ((1-alpha) * delq_attr)

                        if comp_mod_gain > max_gain:
                            max_gain = comp_mod_gain
                            max_index = j
                #print max_gain, max_index
                if max_gain > 0:
                    changeFlag = True
                    #i is moved to max_index
                    dest_comm = node_communitymap[max_index]
                    source_comm = node_communitymap[i]
                    if dest_comm != source_comm:
                        node_communitymap[i] = dest_comm
                        community_nodemap[dest_comm].add(i)
                        community_nodemap[source_comm].discard(i)
                        if len(community_nodemap[source_comm]) == 0:
                            community_nodemap.pop(source_comm, None)
            print "Iteration:",cnt
            cnt+= 1
    except KeyError as e:
        print "Error ==================================="
        print e
        pp.pprint(community_nodemap)
        pp.pprint(node_communitymap)
    if iter_count == 1:
        com_node = phase2(attrs, adj_list, V, E, community_nodemap, node_communitymap, sim_metric)
        return com_node
    return community_nodemap

def phase2(attrs, adj_list, V, E, community_nodemap, node_communitymap, sim_metric):
    length = len(community_nodemap)
    V = range(0, length)
    c = 0
    for key in community_nodemap:
        p2_p1_map[c] = community_nodemap[key]
        c += 1

    adj_list1 = {}
    attrs1 = [0]*len(V)
    #For each cluster
    for key in V:
        #Get all nodes
        nodes = p2_p1_map[key]
        #For each node, get the outgoing edges
        outgoing_edges = []
        #Also for each cluster get the centroid 'attr' list.
        attrs1[key] = [0]*65

        for n in nodes:
            n_outgoing = set(adj_list[n]) - nodes
            outgoing_edges.extend(n_outgoing)
            attrs1[key] = map(add, attrs1[key], map(int,attrs[n]))
        #Set the adjacency list for the cluster.
        adj_list1[key] = outgoing_edges
        div_list = [len(nodes)]*65
        attrs1[key] = map(div, attrs1[key],div_list)



    community_nodemap1 = {}
    node_communitymap1 = {}
    c1 = 0
    #initialize each node to a community
    for node in V:
        community_nodemap1['C'+str(c1)] = set([node])
        node_communitymap1[node] = 'C'+str(c1)
        c1 += 1
    sim_metric1 = {}

    com_node = phase1(attrs1, adj_list1, V, E, community_nodemap1, node_communitymap1, sim_metric1, 2)
    return com_node

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

    V = range(0,324)

    #Create edge list
    E = []
    for node in adj_list.keys():
        for i in adj_list[node]:
            E.append(str(node)+"_"+str(i))

    community_nodemap = {}
    node_communitymap = {}
    c = 0
    #initialize each node to a community
    for node in V:
        community_nodemap['C'+str(c)] = set([node])
        node_communitymap[node] = 'C'+str(c)
        c += 1
    sim_metric = {}
    com_node = phase1(attrs, adj_list, V, E, community_nodemap, node_communitymap, sim_metric, 1)

    pp.pprint(com_node)
    pp.pprint(p2_p1_map)

    if alpha == 1:
        name = "communitites_1.txt"
    elif alpha == 0:
        name = "communitites_0.txt"
    else:
        name = "communitites_5.txt"

    with open(name, 'w') as f:
        for l in com_node:
            papa_string = ''
            for cluster in com_node[l]:
                string = ''
                for node in p2_p1_map[cluster]:
                    string += str(node)+","
                papa_string += string
            papa_string = papa_string[:-1]
            f.write(papa_string)
            f.write("\n")
