from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

pos_list = []
neg_list = []
timestamp_arr = []
def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")

    counts = stream(ssc, pwords, nwords, 100)
    print_vals(counts)
    make_plot(counts)

def print_vals(counts):
    print("\n\n\n")
    i = 0
    while i < len(counts):
        print("--------------------------------------------------------------------------")
        print("Time: ",str(timestamp_arr[i]))
        print("--------------------------------------------------------------------------")
        print(counts[i][0])
        print(counts[i][1])
        print("\n")
        i += 1

def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """

    x0 = np.arange(12)
    x = x0.tolist()

    parr = []
    narr = []
    for elements in counts:
        parr.append(int(elements[0][1]))
        narr.append(int(elements[1][1]))

    line1, = plt.plot(x, parr, marker='o', color = 'b', label='positive')
    line2, = plt.plot(x, narr, marker='o', color = 'g', label='negative')
    #plt.plot(x, parr, 'g-', x, narr, 'b-')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.ylabel('Word Count')
    plt.xlabel('Time Step')
    plt.show()



def load_wordlist(filename):
    """
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    wlist = set(np.genfromtxt(filename,dtype='str'))
    return wlist

def processDStream(rdd, pwords, nwords, counts, t):
    global pos_list, neg_list, timestamp_arr
    pos_list = set(pwords)
    neg_list = set(nwords)
    cnts = rdd.flatMap(processTweet)

    x = cnts.collect()
    y1 = countPositive(x)
    y2 = countNegative(x)
    lst = []
    lst.append(("positive", len(y1)))
    lst.append(("negative", len(y2)))
    timestamp_arr.append(t)

    #counts at every time interval
    print(lst)
    counts.append(lst)
    #print(pos,neg)

def countPositive(arr):
    arr1 = [x.lower() for x in arr]
    print(arr1)
    print(len(arr1))
    y = set(arr1)
    z = pos_list & y
    print(z)
    print(len(z))
    return pos_list & y

def countNegative(arr):
    arr1 = [x.lower() for x in arr]
    y = set(arr1)
    z = neg_list & y
    print(z)
    print(len(z))
    return neg_list & y

def processTweet(tweet):
    return tweet.split(" ")


def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    #tweets.pprint()

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE


    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []

    tweets.foreachRDD(lambda t,rdd: processDStream(rdd, pwords, nwords, counts, t))
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()
