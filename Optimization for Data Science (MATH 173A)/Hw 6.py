#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import random as rand


# In[4]:


training_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_train.csv")


# In[5]:


test_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_test.csv")


# In[6]:


train_9 = ((training_data.loc[training_data['5'] == 9]).iloc[0:500]).to_numpy()
train_4 = ((training_data.loc[training_data['5'] == 4]).iloc[0:500]).to_numpy()

test_9 = ((test_data.loc[test_data['7'] == 9]).iloc[0:500]).to_numpy()
test_4 = ((test_data.loc[test_data['7'] == 4]).iloc[0:500]).to_numpy()

train_9_4 = np.concatenate((train_9, train_4))
test_9_4 = np.concatenate((test_9, test_4))


train_9_4 = np.delete(train_9_4, 0, axis = 1)
test_9_4 = np.delete(test_9_4, 0, axis = 1)

## we can use the same y vector since we have constructed our X the same as the 1s and 0s case


# In[7]:


## now we need to label our x-vectors with appropriate y-values. I have stacked my x-vectors so that the first 500 are nines, 
## and the next 500 are fours. To assign appropriate y-values, all we have to do is concatenate a 500-length vector of 1s
## with a 500 length vector of -1s so long as the concatenation order is aligned with that of the Xs.

train_a = np.ones(500)
train_b = -1*np.ones(500)

# This vector will be our y-vector in all GD trials
train_sign = np.concatenate((train_a, train_b))

## we can use this y-vector for both the training and test data, since they were constructed in the same way. 


# In[8]:


len(train_9_4[1])


# In[9]:


## I was recieving a lot of Math Overflow errors, and I was uncertain whether they were coming from exponential terms that were
## too large, or too small (probably too large). This function breaks the calculation of an exponential dot-product by
## first calculating the individual terms of the product, then multiplying them together in sequence. 

def split_exp_calc (w_t, x, y):
    product = 1
    for i in range(0, len(w_t)):
        product = product * math.exp(w_t[i]*x[i]*y)
    
    return product


# In[10]:


def calc_gradient(w_t, x_vector, y_vector):
    
    v_size = len(x_vector)
    sum = np.zeros(len(w_t), dtype = np.float64)  

    for i in range(0, v_size):
        
        sum += x_vector[i] * (-1*y_vector[i])/(1 + split_exp_calc(w_t, x_vector[i], y_vector[i]))
        
    return (sum/v_size)


# In[11]:


def f_of_w (w_t, x, y):
    
    sum = 0
    for i in range(0, len(x)):
        sum = sum + math.log(1 + split_exp_calc(w_t, x[i], -y[i]))
        
    return sum/(len(x))


# In[12]:


def GD_momentum(w_t, w_t_1, x_vector, y_vector, mu, beta, T):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    count = 0
    F_w_record = []
    
    while count < T:
        
        #calculate new gradient with w_t (stored to check minimality conditions)
        gradient = calc_gradient(w_t, x_vector, y_vector)
        
        # find w_{t+1} 
        w_new = w_t - mu*gradient + beta*(w_t - w_t_1)
        
        # record value of w_{t+1}
        F_w_record.append(f_of_w(w_new, x_vector, y_vector))
        
        # move w_{t-1} forward to point to the old w_t
        w_t_1 = w_t
        
        # move w_t forward to point to the new w_{t+1}
        w_t = w_new
        

        count = count + 1
        
        # if the gradient is small enough, return current w_t and the record of function values
        if la.norm(gradient) < 10**-3 :
            print("broken by gradient condition")
            return w_t, np.asarray(F_w_record)
        
        
    print("broken by trial number")
    return w_t, np.asarray(F_w_record)


# In[29]:


pic_len = len(train_9_4[1])
w_t_momentum = np.zeros(pic_len)
w_t_1_momentum = np.zeros(pic_len)

w_star_6_2_a = GD_momentum(w_t_momentum, w_t_1_momentum, train_9_4, train_sign, 10**-5, 0.9, 200)


# In[92]:


print(f_of_w(w_star_6_2_a[0], train_9_4, train_sign))

plt.plot(w_star_6_2_a[1])


# In[93]:


y_test_6_2a = np.zeros(len(test_9_4))

for i in range (0, len(test_9_4)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_6_2_a[0], test_9_4[i]) < 0):
        y_test_6_2a[i] = -1
    else:
        y_test_6_2a[i] = 1


# In[94]:


mistake_count_6_2a = 0
for i in range (1, len(train_sign)):
    if(train_sign[i] != y_test_6_2a[i]):
        mistake_count_6_2a += 1
        
print(mistake_count_6_2a / len(train_sign))


# In[31]:


pic_len = len(train_9_4[1])
w_t_momentum_b = np.zeros(pic_len)
w_t_1_momentum_b = np.zeros(pic_len)

w_star_6_2b = GD_momentum(w_t_momentum_b, w_t_1_momentum_b, train_9_4, train_sign, 10**-6, 0.95, 200)


# In[32]:


axis = range(0,200)
plt.plot(axis, w_star_6_2b[1])

plt.plot(axis, w_star_6_2_a[1], color = 'r')


# In[33]:


w_star_6_2b[1]


# In[15]:


y_test_6_2b = np.zeros(len(test_9_4))

for i in range (0, len(test_9_4)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_6_2b[0], test_9_4[i]) < 0):
        y_test_6_2b[i] = -1
    else:
        y_test_6_2b[i] = 1


# In[16]:


mistake_count_6_2b = 0
for i in range (1, len(train_sign)):
    if(train_sign[i] != y_test_6_2b[i]):
        mistake_count_6_2b += 1
        
print(mistake_count_6_2b / len(train_sign))


# In[ ]:





# In[99]:


def GD_Nesterov(w_t, w_t_1, x_vector, y_vector, mu, beta, T):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    count = 0
    F_w_record = []
    
    while count < T:
        
        y_t = w_t + beta*(w_t - w_t_1)
        
        #calculate new gradient with w_t (stored to check minimality conditions)
        gradient = calc_gradient(y_t, x_vector, y_vector)
        
        # find w_{t+1} 
        w_new = y_t - mu*gradient
        
        # record value of w_{t+1}
        F_w_record.append(f_of_w(w_new, x_vector, y_vector))
        
        # move w_{t-1} forward to point to the old w_t
        w_t_1 = w_t
        
        # move w_t forward to point to the new w_{t+1}
        w_t = w_new
        
        
        count = count + 1
        
        # if the gradient is small enough, return current w_t and the record of function values
        if la.norm(gradient) < 10**-3 :
            return w_t, np.asarray(F_w_record)
        
        
    return w_t, np.asarray(F_w_record)


# In[100]:


pic_len = len(train_9_4[1])
w_t_Nesterov = np.zeros(pic_len)
w_t_1_Nesterov = np.zeros(pic_len)

w_star_6_3 = GD_Nesterov(w_t_momentum, w_t_1_momentum, train_9_4, train_sign, 10**-5, 0.9, 200)


# In[101]:


plt.plot(w_star_6_3[1])


# In[102]:


y_test_6_3 = np.zeros(len(test_9_4))

for i in range (0, len(test_9_4)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_6_3[0], test_9_4[i]) < 0):
        y_test_6_3[i] = -1
    else:
        y_test_6_3[i] = 1


# In[103]:


mistake_count_6_3 = 0
for i in range (1, len(train_sign)):
    if(train_sign[i] != y_test_6_3[i]):
        mistake_count_6_3 += 1
        
print(mistake_count_6_3 / len(train_sign))


# In[104]:


def F(w, X, y):
    return np.log(np.exp(-y * (X @ w)) + 1).sum() / len(X)


# In[134]:


## Fletcher-Reeves 
def CGD_FR(w_t, x_vector, y_vector, c_1, c_2, T):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    count = 0
    F_w_record = []
    gradient_t = calc_gradient(w_t, x_vector, y_vector)
    p_t = -gradient_t
    
    
    
    while (la.norm(gradient_t) != 0 or count < T):
        
        F_t = F(w_t, x_vector, y_vector)
        
        alpha_t = 1
        
        ## Here we check both Wolfe conditions
        while(F(w_t + alpha_t*p_t, x_vector, y_vector) > F_t + c_1*alpha_t*np.dot(gradient_t, p_t)
             or
             abs(np.dot(calc_gradient(w_t + alpha_t*p_t, x_vector, y_vector), p_t)) > -c_2*np.dot(gradient_t, p_t)):
            
            alpha_t = 0.7*alpha_t
        
        w_t_new = w_t + alpha_t*p_t
        
        gradient_t_new = calc_gradient(w_t_new, x_vector, y_vector)
        
        beta_t_new = np.dot(gradient_t_new, gradient_t_new)/np.dot(gradient_t, gradient_t)
        
        p_t_new = -gradient_t_new + beta_t_new*p_t
        
        F_w_record.append(F(w_t_new, x_vector, y_vector))
        
        # Update the variables needed for the next iteration
        w_t = w_t_new
        gradient_t = gradient_t_new
        p_t = p_t_new

        print(F_w_record[count])
        count = count + 1
        
        #if the gradient is small enough, return current w_t and the record of function values
        if la.norm(gradient_t) < 1 :
            return w_t, np.append(np.asarray(F_w_record),(np.zeros(T-count)))
        
        
    return w_t, np.asarray(F_w_record)


# In[138]:


y_test_6_4a = np.zeros(len(test_9_4))

for i in range (0, len(test_9_4)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_6_4a[0], test_9_4[i]) < 0):
        y_test_6_4a[i] = -1
    else:
        y_test_6_4a[i] = 1


# In[139]:


mistake_count_6_4a = 0
for i in range (1, len(train_sign)):
    if(train_sign[i] != y_test_6_4a[i]):
        mistake_count_6_4a += 1
        
print(mistake_count_6_4a / len(train_sign))


# In[140]:


## Polak-Ribiere
def CGD_PR(w_t, x_vector, y_vector, c_1, c_2, T):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    count = 0
    F_w_record = []
    gradient_t = calc_gradient(w_t, x_vector, y_vector)
    p_t = -gradient_t
    
    
    
    while (la.norm(gradient_t) != 0 or count < T):
        
        F_t = F(w_t, x_vector, y_vector)
        
        alpha_t = 1
        while(F(w_t + alpha_t*p_t, x_vector, y_vector) > F_t + c_1*alpha_t*np.dot(gradient_t, p_t)
             or
             abs(np.dot(calc_gradient(w_t + alpha_t*p_t, x_vector, y_vector), p_t)) > -c_2*np.dot(gradient_t, p_t)):
            
            alpha_t = 0.7*alpha_t
        
        w_t_new = w_t + alpha_t*p_t
        
        gradient_t_new = calc_gradient(w_t_new, x_vector, y_vector)
        
        beta_t_new = np.dot(gradient_t_new, (gradient_t_new - gradient_t) / (la.norm(gradient_t)**2))
        
        p_t_new = -gradient_t_new + beta_t_new*p_t
        
        F_w_record.append(F(w_t_new, x_vector, y_vector))
        
        # Update the variables needed for the next iteration
        w_t = w_t_new
        gradient_t = gradient_t_new
        p_t = p_t_new
        
        
        count = count + 1
        
        # if the gradient is small enough, return current w_t and the record of function values
        if la.norm(gradient_t) < 1 :
            return w_t, np.append(np.asarray(F_w_record),(np.zeros(T-count)))
        
        
    return w_t, np.asarray(F_w_record)


# In[141]:


pic_len = len(train_9_4[1])
w_t_PR = np.zeros(pic_len)

w_star_6_4b = CGD_PR(w_t_PR, train_9_4, train_sign, 0.4, 0.4, 200)


# In[142]:


plt.plot(w_star_6_4b[1])


# In[143]:


y_test_6_4b = np.zeros(len(test_9_4))

for i in range (0, len(test_9_4)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_6_4b[0], test_9_4[i]) < 0):
        y_test_6_4b[i] = -1
    else:
        y_test_6_4b[i] = 1


# In[144]:


mistake_count_6_4b = 0
for i in range (1, len(train_sign)):
    if(train_sign[i] != y_test_6_4b[i]):
        mistake_count_6_4b += 1
        
print(mistake_count_6_4b / len(train_sign))


# In[20]:


axis = range(1, len(w_star_6_2_a))
six_2a_plt = np.empty(len(axis))
six_2b_plt = np.empty(len(axis))
six_3_plt = np.empty(len(axis))
six_4a_plt = np.empty(len(axis))
six_4b_plt = np.empty(len(axis))
for i in range(0, len(axis)):
    six_2a_plt[i] = np.log(w_star_6_2_a[1][i])
    six_2b_plt[i] = np.log(w_star_6_2b[1][i])
    six_3_plt[i]  = np.log(w_star_6_3[1][i])
    six_4a_plt[i] = np.log(w_star_6_4a[1][i])
    six_4b_plt[i] = np.log(w_star_6_4b[1][i])


# In[149]:


plt.plot(axis, six_2a_plt, color = 'b')
plt.plot(axis, six_2b_plt, color = 'g')
plt.plot(axis, six_3_plt, color = 'r')
plt.plot(six_4a_plt, color = 'm')
plt.plot(six_4b_plt, color = 'y')


# In[ ]:





# In[ ]:




