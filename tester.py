import numpy as np

logged = np.array([[1,2,3,4,-6],[1.1,2.2,3.5,4.3,-7.1]]) # (2,5) (number of samples, number of days)
open_price = np.array([100,120,80,50,111])

#taking the mean before undoing the log
mu_before = np.mean(logged,axis=0)
std_before = np.std(logged,axis=0)
mean = ((np.exp(mu_before) * open_price) - open_price)
std = ((np.exp(std_before) * open_price) - open_price)
print(f"before mean = {mean}")

#taking the mean after undoing the log
open_price = np.tile(open_price,(2,1))
unlogged = np.multiply(np.exp(logged),open_price) - open_price
mu_after = np.mean(unlogged,axis=0)
std_after = np.std(unlogged,axis=0)
print(f"after mean = {mu_after}")

print(f"before std = {std}")
print(f"after std = {std_after}")


x = np.zeros((2,3))
y = np.ones((2,5))
print(np.concatenate([x,y],axis =1))
print(np.concatenate([y,x]))