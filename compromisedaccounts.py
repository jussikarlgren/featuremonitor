import os
import re
from logger import logger
import nltk

monitor = True

datadirectory = "/home/jussi/data/recfut/2019.03.cyberattackathon/"
filenamepattern = re.compile(r"fragment_400k.csv")

def getfilelist(resourcedirectory: str, pattern: str):
    filenamelist = []
    for filenamecandidate in os.listdir(resourcedirectory):
        if pattern.match(filenamecandidate):
            logger(filenamecandidate, monitor)
            filenamelist.append(os.path.join(resourcedirectory,filenamecandidate))
    logger(filenamelist, monitor)
    return sorted(filenamelist)

interestingnpattern = re.compile(r".*accounts.*")

claimterms = [("claims", "VBZ"), ("claimed", "VBD"), ("claimed", "VBN")]

filename = "/home/jussi/data/recfut/2019.03.cyberattackathon/fragment_100k.csv"
with open(filename, "rt") as f:
    for line in f:
        ll = line.strip()
        if ll and interestingnpattern.match(ll):
            words = nltk.word_tokenize(ll)
            poswords = nltk.pos_tag(words)
            poses = [a[1] for a in poswords]
            #if "VBD" in poses or "VBP" in poses or "VBN" in poses:
            #    pass  #                print(words)
            #else:
            if set(claimterms) & set(poswords):
                print("*** " + ll)
                print(poswords)
