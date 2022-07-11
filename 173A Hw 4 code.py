#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import random as rand


# In[3]:


training_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_train.csv")
test_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_test.csv")


# In[4]:


x_nines_tr = ((training_data.loc[training_data['5'] == 9]).iloc[0:500]).to_numpy()
x_fours_tr = ((training_data.loc[training_data['5'] == 4]).iloc[0:500]).to_numpy()

x_nines_te = ((test_data.loc[test_data['7'] == 9]).iloc[0:500]).to_numpy()
x_fours_te = ((test_data.loc[test_data['7'] == 4]).iloc[0:500]).to_numpy()

x_te_all_nines_and_fours = np.concatenate((x_nines_te, x_fours_te))
x_tr_all_nines_and_fours = np.concatenate((x_nines_tr, x_fours_tr))


x_tr_all_nines_and_fours = np.delete(x_tr_all_nines_and_fours, 0, axis = 1)
x_te_all_nines_and_fours = np.delete(x_te_all_nines_and_fours, 0, axis = 1)


# In[5]:


y_tr_ones = np.ones(500)
y_tr_zeros = -1*np.ones(500)
y_tr_all = np.concatenate((y_tr_ones, y_tr_zeros))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


def split_exp_calc (w_t, x, y):
    product = 1
    for i in range(0, len(w_t)):
        product = product * math.exp(w_t[i]*x[i]*y)
    
    return product


# In[7]:


def calc_gradient(w_t, x_vector, y_vector):
    
    v_size = len(x_vector)
    sum = np.zeros(len(w_t), dtype = np.float64)  

    for i in range(0, v_size):
        
        sum += x_vector[i] * (-1*y_vector[i])/(1 + split_exp_calc(w_t, x_vector[i], y_vector[i]))
        
    return (sum/v_size)


# In[8]:


def f_of_w (w_t, x, y):
    
    sum = 0
    for i in range(0, len(x)):
        sum = sum + math.log(1 + split_exp_calc(w_t, x[i], -y[i]))
        
    return sum/(len(x))


# In[9]:


def descend_original(w_t, x_vector, y_vector, mu, count, ceiling):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    gradient = calc_gradient(w_t, x_vector, y_vector)
    F_w_record = np.empty(0)
    
    while count < ceiling:
        w_t -= mu*gradient
        gradient = calc_gradient(w_t, x_vector, y_vector)
        
        F_w_record = np.append(F_w_record, f_of_w(w_t, x_vector, y_vector))
        
     
        count = count + 1
        
        if la.norm(gradient) < 10-3 :
            return w_t, F_w_record
        
        
    return w_t, F_w_record


# In[ ]:





# In[ ]:





# In[10]:


def gradient_3a (del_f):
    sum = 0
    sign_f = np.ones(len(del_f))
    
    for i in range (1, len(del_f)):
        sum += abs(del_f[i])
        if(del_f[i] < 0):
            sign_f[i] = -1
        
    return (-sum * sign_f)


# In[11]:


def descend_3a(w_t, x_vector, y_vector, mu, count, ceiling):
    
    
    gradient = calc_gradient(w_t, x_vector, y_vector)
    F_w_record = np.empty(0)
    
    while count < ceiling:
        
        w_t += mu* gradient_3a(gradient)
        
        gradient = calc_gradient(w_t, x_vector, y_vector)
        
        F_w_record = np.append(F_w_record, f_of_w(w_t, x_vector, y_vector))
        
        count = count + 1
        
        if la.norm(gradient) < 10-3 :
            return w_t, F_w_record
        
        
    return w_t, F_w_record


# In[19]:


w_test_9_4 = np.zeros(784)
w_star_a = descend_3a(w_test_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-8, 0, 200)


# In[64]:


x_axis_3a = np.array(range(0,len(w_star_a[1])))
plt.plot(x_axis_3a, w_star_a[1])


# In[57]:


w_star_a[1][299]


# In[52]:


print(w_star_a[0])


# In[58]:


## assign +1 or -1 to each of the test X values based on our idea w_t

y_vec_test_nines_and_fours = np.zeros(len(x_te_all_nines_and_fours))

for i in range (0, len(x_te_all_nines_and_fours)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_a[0], x_te_all_nines_and_fours[i]) < 0):
        y_vec_test_nines_and_fours[i] = -1
    else:
        y_vec_test_nines_and_fours[i] = 1


# In[59]:


## now we can compare the generated vector with the known identity vector y_tr_all
mistake_count_9_4 = 0
for i in range (1, len(y_vec_test_nines_and_fours)):
    if(y_vec_test_nines_and_fours[i] != y_tr_all[i]):
        mistake_count_9_4 += 1
        
print(mistake_count_9_4 / len(y_vec_test_nines_and_fours))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


def gradient_3b (del_f):
    
    ## This vector will be our direction vector for the p^t calculation
    pos_vec = np.zeros(len(del_f))
    current_max = np.amax(abs(del_f))
    
    ## this is a dummy variabled used to track the assignment of the direction vector
    assigned = 1
    
    for i in range (1, len(del_f)):

        if(assigned == 1):
            if (del_f[i] == current_max):
                pos_vec[i] = 1
                assigned = 0
            elif(del_f[i] == -current_max):
                pos_vec[i] = -1
                assigned = 0
                
    return (-current_max * pos_vec)
        


# In[13]:


def descend_3b(w_t, x_vector, y_vector, mu, count, ceiling):
        
    
    gradient = calc_gradient(w_t, x_vector, y_vector)
    F_w_record = np.empty(0)
    
    while count < ceiling:
        
        w_t += mu* gradient_3b(gradient)
        
        gradient = calc_gradient(w_t, x_vector, y_vector)
        
        F_w_record = np.append(F_w_record, f_of_w(w_t, x_vector, y_vector))
        
        count = count + 1
        
        if la.norm(gradient) < 10-3 :
            return w_t, F_w_record
        
        
    return w_t, F_w_record


# In[18]:


w_test_9_4 = np.zeros(784)
w_star_b = descend_3b(w_test_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-4, 0, 200)


# In[68]:


x_axis_3b = np.array(range(0,len(w_star_b[1])))
plt.plot(x_axis_3b, w_star_b[1])


# In[45]:


print(w_star_b[0])


# In[48]:


## assign +1 or -1 to each of the test X values based on our idea w_t

y_vec_test_nines_and_fours_b = np.zeros(len(x_te_all_nines_and_fours))

for i in range (0, len(x_te_all_nines_and_fours)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(w_star_b[0], x_te_all_nines_and_fours[i]) < 0):
        y_vec_test_nines_and_fours_b[i] = -1
    else:
        y_vec_test_nines_and_fours_b[i] = 1


# In[49]:


## now we can compare the generated vector with the known identity vector y_tr_all
mistake_count_9_4 = 0
for i in range (1, len(y_vec_test_nines_and_fours_b)):
    if(y_vec_test_nines_and_fours_b[i] != y_tr_all[i]):
        mistake_count_9_4 += 1
        
print(mistake_count_9_4 / len(y_vec_test_nines_and_fours_b))


# In[72]:


w_test_9_4 = np.zeros(784)
x_star_9_4 = descend_original(w_test_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-6, 0, 200)


# In[73]:


plt.plot(x_axis_3a, w_star_a[1], color = 'r', label = '3a plot')
plt.plot(x_axis_3a, w_star_b[1], color = 'g', label = '3b plot')
plt.plot(x_axis_3a, x_star_9_4[1], color='b', label='original GD')


# In[15]:


w_test_9_4 = np.zeros(784)
w_star_3c_a = descend_3a(w_test_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-6, 0, 200)
w_test_9_4 = np.zeros(784)
w_star_3c_b = descend_3b(w_test_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-6, 0, 200)


# In[23]:


x_axis_3 = np.array(range(0,len(w_star_a[1])))
plt.plot(x_axis_3, w_star_a[1], color = 'r', label = '3a plot')
plt.plot(x_axis_3, w_star_b[1], color = 'g', label = '3b plot')

plt.plot(x_axis_3, w_star_3c_b[1], color = 'b', label = '3(c) b plot')


# In[24]:


plt.plot(x_axis_3, w_star_3c_a[1], color = 'purple', label = '3(c) a plot')


# In[ ]:




