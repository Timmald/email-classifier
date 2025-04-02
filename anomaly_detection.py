import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

fake_ham = np.random.rand(10000,500)

fake_spam = np.random.rand(10000,500)
fake_spam = np.concatenate([fake_spam,np.ones((10000,1))],axis=-1)

train,test_ham = train_test_split(fake_ham,test_size=.2,train_size=.8)
test_ham = np.concatenate([test_ham,np.zeros((2000,1))],axis=-1)
_,test_spam = train_test_split(fake_spam,test_size=.2,train_size=.8)

test = np.concatenate([test_ham,test_spam])

testzip = zip(test[:,:-1],test[:,-1])#(item,label) format
correct = 0
mean_eucs = []
for item,label in testzip:
    mean_eucs.append(np.mean(np.linalg.norm(np.tile(item,(len(train),1))-train)))

thresh50 = np.median(mean_eucs)
print(thresh50)
i = 0
for item,label in zip(test[:,:-1],test[:,-1]):
    if mean_eucs[i] > thresh50:
        pred = 1
    else:
        pred = 0
    correct += int(pred==label)
    i+=1
print(correct)
print(correct/len(test))



