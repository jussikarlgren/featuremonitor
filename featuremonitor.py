import nltk
from collections import Counter
import json
import csv
import re
import os
import numpy as np
from logger import logger
import khi2
# import pickle

monitor = True
debug = True
error = True

# chk burstiness in output
# chk burstiness in training mat
# regularise naturally thru instruction, by offering up more data about what seems to be interesting,
# but do not claim that the new materials are positive or negative examples, just neutral!

defaultresourcedirectory = "/home/jussi/data/incident/featuremonitor/"

def getCSVdata(csvfile):
    """opens a csv file, reads it in and returns a 2 dimensional list with the data."""
    try:
        with open(defaultresourcedirectory + "analyzed_0.7.csv") as ff:
            reader = csv.reader(ff)
            datalist=[]
            for row in reader:
                rowlist=[]
                for col in row:
                    rowlist.append(col)
                datalist.append(rowlist)
            return datalist
    except:
        print("Could not open file {}".format(csvfile))


def getfilelist(resourcedirectory=defaultresourcedirectory, fileexpression=r"2018-.*"):
    pattern = re.compile(fileexpression)
    filenamelist = []
    for filenamecandidate in os.listdir(resourcedirectory):
        if pattern.match(filenamecandidate):
            filenamelist.append(os.path.join(resourcedirectory,filenamecandidate))
    return sorted(filenamelist)


def burstiness(word:str, specificvocabs:dict):
    vec = np.zeros(len(specificvocabs))
    i = 0
    for vv in specificvocabs:
        try:
            if specificvocabs[vv][word] > 0:
                i += 1
        except KeyError:
            pass
    return i
    # vec[i] = specificvocabs[vv][word]
    #    i += 1
    #return(vec.std())


def rank(word:str, vocab:Counter):
    order = [z[0] for z in sorted(vocab.items(), key=lambda x: x[1], reverse=True)]
    try:
        r = order.index(word)
    except:
        r = len(order)
    return r


def comb(c:Counter, threshold=3):
    a = Counter()
    for i in c:
        if c[i] >= threshold:
            a[i] = c[i]
    return a


def wordkhi2(w:str, loglevel:bool=False):
    a = documentfrequency[w]
    c = osevendocumentfrequency[w]
    b = antaldokument - a
    d = antaloseven - c
    x = np.array(((a, b), (c, d)))
    logger("{}".format(w), loglevel)
    fieldlength = len(str(a))
    logger("{:{}}\t{:{}}\t|\t{:{}} ".format(a, fieldlength, b, fieldlength, antaldokument, fieldlength), loglevel)
    logger("{:{}}\t{:{}}\t|\t{:{}} ".format(c, fieldlength, d, fieldlength, antaloseven, fieldlength), loglevel)
    logger("-" * fieldlength + "\t" + "-" * fieldlength + "\t+\t" + "-" * fieldlength, loglevel)
    logger("{:{}}\t{:{}}\t|\t{:{}} ".format(a+c, fieldlength,b+d, fieldlength,
                                            antaldokument + antaloseven, fieldlength), loglevel)
    k = khi2.khi2(x)
    exp = (a + b) * (a + c) / (a + b + c + d)
    if exp < a:
        k = -k
    logger("khi2 = {}".format(k), loglevel)
    return k



vocab = Counter()
documentfrequency = Counter()
dailyvocab = {}
topicvocab = {}
sourcetopicvocab = {}
antaldokument = 0
resourcedirectory = defaultresourcedirectory
outputdirectory = defaultresourcedirectory + "scratch/"
fileexpression = r"2018-.*[0-9]$"


def readthefiles():
    global antaldokument, vocab, hapaxfilteredvocab, documentfrequency, dailyvocab, topicvocab, sourcetopicvocab
    global resourcedirectory, outputdirectory, fileexpression
    filelist = getfilelist(resourcedirectory, fileexpression)
    logger(str(filelist), monitor)
    vocab = Counter()
    documentfrequency = Counter()
    dailyvocab = {}
    topicvocab = {}
    sourcetopicvocab = {}
    antaldokument = 0
    antalfiler = 0
    feldokument = 0
    words = "placeholder"
    t2 = "zip"
    f = ""
    for dayfile in filelist:
        antalfiler += 1
        dailyvocab[dayfile] = Counter()
        try:
            j = json.loads(open(dayfile).read())
            for i in j["instances"]:
                try:
                    if i["document"]["language"] == "eng":
                        try:
                            t1s = i["attributes"]["topics"]
                        except:
                            t1s = ["natch"]
                        for t1 in t1s:
                            if t1 not in topicvocab:
                                topicvocab[t1] = Counter()
                        try:
                            t2 = i["document"]["sourceId"]["topic"]
                        except:
                            t2 = "zip"
                        if t2 not in sourcetopicvocab:
                            sourcetopicvocab[t2] = Counter()
                        try:
                            words = nltk.word_tokenize(i["fragment"].lower())
                            antaldokument += 1
                        except KeyError:
                            words = []
                            feldokument += 1
                            logger("KeyError with {} in {}".format(eeee.args, i), debug)
                        vocab.update(words)
                        wordset = set(words)
                        documentfrequency.update(wordset)
                        dailyvocab[dayfile].update(wordset)
                        for t1 in t1s:
                            topicvocab[t1].update(wordset)
                        sourcetopicvocab[t2].update(wordset)
                    else:
                        pass  # non-english
                except Exception as eeee:
                    feldokument += 1
                    logger("Error {} with {} after {}, {} {}".format(type(eeee), eeee.args, t2, words, i), debug)
        except Exception as ee:
            logger("File error {} for {}".format(ee, f), monitor)
        logger("Antal filer: {}; just klar med {}; antal dokument: {}; antal feldokument: {}".format(antalfiler, dayfile,
                                                                                                     antaldokument,
                                                                                                     feldokument),
               monitor)
        #    for d in dailyvocab:
        #        dailyvocab[d] = comb(dailyvocab[d])
        #    for d in sourcetopicvocab:
        #        sourcetopicvocab[d] = comb(sourcetopicvocab[d])
        #    for d in topicvocab:
        #        topicvocab[d] = comb(topicvocab[d])
        with open(outputdirectory + "vocab{}.json".format(len(filelist)), "w+") as f:
            f.write(json.dumps(vocab))
        with open(outputdirectory + "dailyvocab{}.json".format(len(filelist)), "w+") as f:
            f.write(json.dumps(dailyvocab))
        with open(outputdirectory + "sourcetopicvocab{}.json".format(len(filelist)), "w+") as f:
            f.write(json.dumps(sourcetopicvocab))
        with open(outputdirectory + "topicvocab{}.json".format(len(filelist)), "w+") as f:
            f.write(json.dumps(topicvocab))

    # do katz stats per cluster_id and per date


readdata = True
if __name__ == '__main__':
    projectdirectory = defaultresourcedirectory
    if readdata:
        readthefiles()
    else:
        datatag = "9"
        with open(projectdirectory + "vocab{}.json".format(datatag), "r+") as f1:
            vocab = json.loads(f1.read())
        with open(projectdirectory + "dailyvocab{}.json".format(datatag), "r+") as f2:
            dailyvocab = json.loads(f2.read())
            for item in dailyvocab:
                documentfrequency.update(dailyvocab[item])
                antaldokument += 1
        with open(projectdirectory + "sourcetopicvocab{}.json".format(datatag), "r+") as f3:
            sourcetopicvocab = json.loads(f3.read())
        with open(projectdirectory + "topicvocab{}.json".format(datatag), "r+") as f4:
            topicvocab = json.loads(f4.read())
    hapaxfilteredvocab = comb(vocab)
    oseven = Counter()
    antaloseven = 0
    osevendocumentfrequency = Counter()
    with open(projectdirectory + "analyzed_0.7.csv", "r+") as ff:
        r = csv.reader(ff)
        for rr in r:
            antaloseven += 1
            f0 = nltk.word_tokenize(rr[0].lower())
            oseven.update(f0)
            f0set = set(f0)
            osevendocumentfrequency.update(f0set)

    #probes = ["suicide", "killing", "incident", "bomb", "christmas", "bomb", "unknown" ]
    #compare
    #for w in probes:
    #    print(w)
    #    a = vocab[w]
    #    c = oseven[w]
    #    b = len(vocab) - a
    #    d = len(oseven) - c
    #    x = np.array(((a,c),(b,d)))
    #    k = khi2.khi2(x)
    #    print("main: {}/{}\tevent: {}/{}\tkhi2: {}".format(rank(w, vocab), len(vocab), rank(w, oseven), len(oseven), k))
    #    print(burstiness(w, vocab, dailyvocab))
    #    print(burstiness(w, vocab, topicvocab))
    #    print(burstiness(w, vocab, sourcetopicvocab))

    khi2score = {}
    for w in oseven:
        k = wordkhi2(w)
        khi2score[w] = k
    toppantal = 100
    bottantal = 10
    kest = sorted(khi2score.items(), key=lambda x: x[1], reverse=True)
    best = [z[0] for z in kest][:toppantal]
    semst = [z[0] for z in kest][-bottantal:]
    for wb in best + semst:
        print("{}\t{}\t{}\t{}/{}\t{}\t{}/{}\t{}\t{}/{}\t{}\t{}\t{}/{}\t{}\t{}/{}\t{}\t{}\t{}\t{}".format(
            wb,
            antaldokument,
            vocab[wb],
            rank(wb, vocab), len(vocab),
            hapaxfilteredvocab[wb],
            rank(wb, hapaxfilteredvocab), len(hapaxfilteredvocab),
            documentfrequency[wb],
            rank(wb, documentfrequency), len(documentfrequency),
            antaloseven,
            oseven[wb],
            rank(wb, oseven), len(oseven),
            osevendocumentfrequency[wb],
            rank(wb, osevendocumentfrequency), len(osevendocumentfrequency),
            khi2score[wb],
            burstiness(wb, dailyvocab),
            burstiness(wb, topicvocab),
            burstiness(wb, sourcetopicvocab)
#            burstiness(wb, documentfrequency)
        ))

