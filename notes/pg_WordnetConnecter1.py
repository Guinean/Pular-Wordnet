import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from array import array
import re

with open('FulaSenseEnglish.txt', 'r', encoding='utf-8') as f:
    read_data = f.readlines()

with open('FulaPOSTags.txt', 'r', encoding='utf-8') as g:
    pos_tags = g.readlines()

def posTag(tag, tags): #determine if tag is in tags
    for t in tags:
        if (tag == t):
            return 1
    return 0

verbsID = []
adjsID = []
advsID = []
for line in pos_tags: #assigning the fula ids to Adv, Verb, Adj groups/lists
    fulaID = line.split('\t')[0]
    tags = line[:-1].split('\t')[1:]
    for tag in tags:
        if (fulaID not in verbsID and posTag(tag,["v.", "v.av", "v.i.", "v.mv", "v.pv", "v.t."]) == 1):
            verbsID.append(fulaID)
        elif (fulaID not in adjsID and posTag(tag,["adj", "adj."]) == 1):
            adjsID.append(fulaID)
        elif (fulaID not in advsID and posTag(tag,["adv", "adv."]) == 1):
            advsID.append(fulaID)

synsetFile = open('WordnetEnglish1.txt', 'w', encoding='utf-8')
unconnectedFile = open('WordnetUnconnected1.txt', 'w', encoding='utf-8')
matched = 0
matched100 = 0
unmatched = 0

for line in read_data:
    if (len(line.split('\t')) > 1): #filter blank lines
        if (len(line.split('\t')) > 2): #filter entries that have content in the gloss, (not blank gloss)
            defs = line.split('\t')[2]
        else:
            defs = [] #no gloss given
        fulaID = line.split('\t')[0]
        senseID = line.split('\t')[1]
        while (defs.find('(') >= 0): #glosses with a () have/are specific notes on senses, largly, and are much more vauge in feel. These seem to be being removed?
            if (defs.find(')') < 0):
                defs = defs[:defs.find('(')]
            elif (defs.find(')') < defs.find('(')):
                defs = defs[defs.find(')')+1:]
            else:
                defs = defs[:defs.find('(')] + defs[defs.find(')')+1:]
        if (defs != []):
            defs = defs[:-1].split(', ')
        overall = []
        senses = []
        posID = wn.NOUN
        if (fulaID in verbsID):
            posID = wn.VERB
        elif (fulaID in adjsID):
            posID = wn.ADJ
        elif (fulaID in advsID):
            posID = wn.ADV
        for x in range(len(defs)):
            defs[x] = defs[x].split(' ')
            defs[x] = [y for y in defs[x] if (len(y) > 0) and (len(defs[x]) == 1 or y not in stopwords.words('english'))]
            union = wn.synsets('_'.join(defs[x]), pos = posID)
            if (not union): #if not a perfect match to an entry in the corpus, then try addnitional processing to find one
                wordSenses = []
                for y in range(len(defs[x])):
                    sense = wn.synsets(defs[x][y], pos = posID)
                    if (sense):
                        wordSenses.append(sense)
                if (wordSenses):
                    for x in range(len(wordSenses[0])):
                        found = True
                        word = wordSenses[0][x]
                        for y in range(len(wordSenses)):
                            if (word not in wordSenses[y]):
                                found = False
                        if (found):
                            union.append(word)
            if (union): #is perfect match, add it.
                senses.append(union)
        if (senses): #if a sense was found
            for x in range(len(senses[0])):#some logic based off wordnet structure
                found = True
                word = senses[0][x]
                for y in range(len(senses)):
                    if (word not in senses[y]):
                        found = False
                if (found):
                    overall.append(word)
        if (overall): #scoring
            score = 1
            if (len(senses) == 1):
                score = 0.8/len(overall)
            elif (len(overall) > 1):
                score = 1.0/len(overall)
            prt = "{}\t{}\t{}".format(fulaID, senseID, score)
            if (score == 1):
                matched100 = matched100 + 1
            matched = matched + 1
            for x in overall:
                prt = prt + "\t{}".format(x.name())
            print(prt, file = synsetFile)
        else:
            unmatched = unmatched + 1
            print(line[:-1], file = unconnectedFile)

synsetFile.close()
unconnectedFile.close()
print('Done with {} matches and {} unconnected entries, of which {} have 1.0 confidence score.'.format(matched, unmatched, matched100))

