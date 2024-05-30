from math import factorial
from collections import Counter

def getNumPermutations(word):
    numerator = factorial(len(word))
    wordDict = Counter(word)
    denominator = 1
    for value in wordDict.values():
        denominator *= factorial(value)
    return numerator//denominator

word = "mississippi"
print(getNumPermutations(word))
    

