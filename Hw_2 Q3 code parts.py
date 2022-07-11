#!/usr/bin/env python
# coding: utf-8

# In[16]:


file = open('D:\\a_School Things\\Super Senior (Fall 2021)\\Math 173A\\myFile (1).txt')
for x in file:    print(x)


# In[2]:


myfile = 'D:\\a_School Things\\Super Senior (Fall 2021)\\Math 173A\\myFile (1).txt'


# In[3]:


print(len(x))


# In[4]:


import numpy as np
from numpy import linalg as LA


# In[5]:


test_array = np.loadtxt(myfile, delimiter = ',')
test_array


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# In[58]:


plt.plot(test_array[0], test_array[1], 'o', color='black')


# In[6]:


test_array[0][1]
test_array[1][1]


# In[184]:


test_array.shape


# In[8]:


u_arr = np.square(test_array[0])
u_arr


# In[9]:


v_arr = np.square(test_array[1])


# In[10]:


A_matrix_trans = np.array([u_arr, v_arr])
A_matrix_trans.shape


# In[11]:


A_matrix = A_matrix_trans.T
A_matrix.shape[1]


# In[99]:


A_matrix_trans


# In[19]:


2*(np.dot(A_matrix_trans, np.dot(A_matrix, [1,2])) - np.ones(2).T)
x = [3, 4]


# In[194]:


def descend (x_t, A_mtx, A_mtx_t, mu, ct, runs):
    
    x_t_next = x_t - mu*2*(np.dot(A_mtx_t, (np.dot(A_mtx, x_t) - np.ones(A_mtx_t.shape[1]).T)))
    ct_next = ct + 1
    
    if LA.norm(x_t_next - x_t) <= pow(10, -6) :
        
        print("gradient of x_t")
        print(np.dot(A_mtx_t, (np.dot(A_mtx, x_t) - np.ones(A_mtx_t.shape[1]).T)))
        
        print("value of x_t at stop")
        print(x_t_next)
        return x_t_next
        
    if ct_next > runs:
       
        print(x_t)
        
        return x_t
    
    else:
        return descend(x_t_next, A_mtx, A_mtx_t, mu, ct_next, runs)
    
    
    
    


# In[ ]:





# In[195]:


u_arr = np.square(test_array[0])
v_arr = np.square(test_array[1])


# In[196]:


A_matrix_trans = np.array([u_arr, v_arr])

A_matrix = A_matrix_trans.T

A_matrix_trans.shape


# In[197]:


AtA = np.dot(A_matrix_trans, A_matrix)
mu_1 = 1/(2*LA.norm(AtA))
mu_1


# In[198]:


x_1 = np.array([1,1])


a_trial = descend(x_1, A_matrix, A_matrix_trans, mu_1, 0, pow(10, 3))


# In[166]:


type(a_trial)


# In[157]:


def f_of_a (a, A_mtx_t):
    
    val = np.dot(A_mtx, a) - np.ones(A_mtx_t.shape[1])
    
    return(LA.norm(val, val))


# In[175]:


val = np.dot(A_mtx, a_trial) - np.ones(A_mtx.shape[0])
np.dot(val.T, val)


# In[186]:


x_2 = np.array([1,2])
b_trial = descend(x_2, test_array.T, test_array, mu_1, 0, pow(10, 4))


# In[187]:


val = np.dot(test_array.T, b_trial) - np.ones(A_mtx.shape[0])
np.dot(val.T, val)


# In[57]:


x_t = np.array([1,2])
A_mtx = A_matrix
A_mtx_t = A_matrix_trans

x_t_next = x_t + 2*(np.dot(A_mtx_t, np.dot(A_mtx, x_t)) - np.ones(len(x_t)).T)

LA.norm(x_t_next - x_t)


# In[105]:



num_trials = 10^9
gradient = 2*(np.dot(A_matrix_transpose, np.dot(A_matrix, a_t) - np.ones(test_array[0].size).T)


# In[ ]:





# In[ ]:


def descend (x_t, A_mtx, A_mtx_t, mu, ct, runs):
    
    x_t_next = x_t - mu*2*(np.dot(A_mtx_t, (np.dot(A_mtx, x_t) - np.ones(A_mtx_t.shape[1]).T)))
    ct_next = ct + 1
    
    if LA.norm(x_t_next - x_t) <= pow(10, -6) :
        
        print("gradient of x_t")
        print(np.dot(A_mtx_t, (np.dot(A_mtx, x_t) - np.ones(A_mtx_t.shape[1]).T)))
        
        print("value of x_t at stop")
        print(x_t_next)
        return x_t_next
        
    if ct_next > runs:
       
        print(x_t)
        
        return x_t
    
    else:
        return descend(x_t_next, A_mtx, A_mtx_t, mu, ct_next, runs)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




