import pandas as pd
import pickle
file1 = "policy_p1"
file2 = "policy_p2"
fr1 = open(file1, 'rb')
fr2 = open(file2, 'rb')

qvals_p1 = {}
qvals_p1 = pickle.load(fr1)
qvals_p2 = {}
qvals_p2 = pickle.load(fr2)
dfp1 = pd.DataFrame(qvals_p1, index = [0]).T
#print(qvals_p1)
dfp2 = pd.DataFrame(qvals_p2, index = [0]).T

dfp1.to_csv('p1policy.csv')
dfp2.to_csv('p2policy.csv')



