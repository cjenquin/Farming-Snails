#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import random as rand


# In[274]:


def gradient_5_3 (x_1, x_2): 
    return  np.array([(800*(x_1**3 - x_2*x_1) + 2*x_1 - 2), (400*(x_2 - x_1**2))])


# In[275]:


def hessian_5_3 (x_1, x_2):
    hessian = np.array([[(2400*x_1**2 - 800*x_2 + 2), (-1* 800*x_1)], [(-1 * 800*x_1), 400]])

    return hessian


# In[277]:


def inverse_hessian_5_3 (x_1, x_2):
    inv_hessian = np.array([[400, (800*x_1)], [(800*x_1), (2400*x_1**2 - 800*x_2 + 2)]])
    determinant = (3.2*10**5)*(x_1**2 - x_2 + 0.0025)
    
    return inv_hessian/determinant


# In[279]:


def f_x (x_t):
    
    return (200*(x_t[1] - x_t[0]**2)**2 + (1 - x_t[0])**2)


# In[421]:


def Newtons_5_3 (x_0, ceiling):
    
    x_t = x_0
    x_1_record = []
    x_2_record = []
    f_x_t_record = []
    count = 0
    
    while (count <= ceiling):
        
        x_1_record.append(x_t[0])
        x_2_record.append(x_t[1])
        f_x_t_record.append(f_x(x_t))
        
        x_t = x_t - np.dot(inverse_hessian_5_3(x_t[0], x_t[1]), gradient_5_3(x_t[0], x_t[1]))
        count += 1
    
    
   
    x_1_record = np.asarray(x_1_record)
    x_2_record = np.asarray(x_2_record)
    f_x_t_record = np.asarray(f_x_t_record)
    
    return [x_t, x_1_record, x_2_record, f_x_t_record]


# In[434]:


x_star_5_3a = Newtons_5_3([0,0], 1000)


# In[307]:


len(x_star_5_3a[1])


# In[308]:


x_1s = np.empty(0)
x_2s = np.empty(0)


for i in range (0, 101 ):
    x_1s = np.append(x_1s, x_star_5_3a[1][2*i])
    x_2s = np.append(x_2s, x_star_5_3a[1][2*(i-1) + 1])


# In[ ]:





# In[423]:


plt.plot(x_star_5_3a[1], x_star_5_3a[2])


# In[424]:


plt.plot(x_star_5_3a[3])


# In[269]:


log_5_3a = np.empty(0)
for i in x_star_5_3a[2]:
    log_5_3a = np.append(log_5_3a, math.log10(abs(i)))


# In[313]:


plt.plot(log_5_3a)


# In[ ]:





# In[453]:


def gradient_descent_5_3 (x_0, ceiling):
    
    x_t = x_0
    x_1_record = []
    x_2_record = []
    f_x_t_record = []
    count = 0
    
    while (count <= ceiling):
        
        x_1_record.append(x_t[0])
        x_2_record.append(x_t[1])
        f_x_t_record.append(f_x(x_t))

        
        x_t -= 10**-3 * gradient_5_3(x_t[0], x_t[1])
        
        print(x_t)
        count += 1
        
    x_1_record = np.asarray(x_1_record)
    x_2_record = np.asarray(x_2_record)
    f_x_t_record = np.asarray(f_x_t_record)   
    
    return [x_t, x_1_record, x_2_record, f_x_t_record]


# In[454]:


x_star_5_3b = gradient_descent_5_3([0,0], 1000)


# In[455]:


plt.plot(x_star_5_3b[3])


# In[351]:


log_5_3b = np.empty(0)
for i in x_star_5_3b[3]:
    log_5_3b = np.append(log_5_3b, math.log10(abs(i)))


# In[352]:


plt.plot(log_5_3b)


# In[427]:


plt.plot(x_star_5_3b[2])


# In[428]:


plt.plot(x_star_5_3b[1])


# In[429]:


x_star_5_3b[0]


# In[ ]:





# In[ ]:





# In[456]:


def BTLS_GD (x_0, ceiling, gamma, beta, mu_base):
    
    x_t = x_0
    
    ## begin vectors for capturing the history of [x_1, x_2]_t and f(x_t)
    x_1_record = []
    x_2_record = []
    f_x_t_record = []
    
    count = 0
    
    while (count <= ceiling):
        
        ## calculate gradient, hessian, and hessian 2-norm for the prior x_t value
        gradient = gradient_5_3(x_t[0], x_t[1])
        
        
        ## reset mu to base 
        mu = mu_base
        
        ## check values of mu until the Armijo condition is met
        while (abs(f_x(x_t - mu*gradient)) > abs(f_x(x_t) - mu*gamma*la.norm(gradient))):
            mu = beta*mu
        print(mu)
           
            
        ## update and print x_t
        x_t -= mu*gradient
        print(x_t)

        
        
        ## record x_t information
        x_1_record.append(x_t[0])
        x_2_record.append(x_t[1])
        f_x_t_record.append(f_x(x_t))
        
        count += 1
        
    x_1_record = np.asarray(x_1_record)
    x_2_record = np.asarray(x_2_record)
    f_x_t_record = np.asarray(f_x_t_record)   
    
    return [x_t, x_1_record, x_2_record, f_x_t_record]


# In[457]:


x_star_5_3c = BTLS_GD([0., 0.], 1000, 0.7, 0.5, 1)


# In[418]:


plt.plot(x_star_5_3c[1])


# In[419]:


plt.plot(x_star_5_3c[2])


# In[431]:


plt.plot(x_star_5_3c[3])


# In[432]:


x_star_5_3c[0]


# In[463]:


axis = range(0, len(x_star_5_3a[3]))


# In[ ]:





# In[458]:


norm_difference_Newtons = []

for i in range (0, axis):
    norm_difference_Newtons.append( math.sqrt( ((x_star_5_3a[1][i] - 1)**2 + (x_star_5_3a[2][i] - 1)**2) ))
    
    
norm_difference_Newtons = np.asarray(norm_difference_Newtons)


# In[459]:


norm_difference_GD = []

for i in range (0, axis):
    norm_difference_GD.append( math.sqrt( ((x_star_5_3b[1][i] - 1)**2 + (x_star_5_3b[2][i] - 1)**2) ))
    
norm_difference_GD = np.asarray(norm_difference_GD)


# In[460]:


norm_difference_BTLS = []

for i in range (0, axis):
    norm_difference_BTLS.append( math.sqrt( ((x_star_5_3c[1][i] - 1)**2 + (x_star_5_3c[2][i] - 1)**2) ))

norm_difference_BTLS = np.asarray(norm_difference_BTLS)


# In[464]:


plt.plot(axis, norm_difference_Newtons, color = 'b')
plt.plot(axis, norm_difference_GD, color = 'g')
plt.plot(axis, norm_difference_BTLS, color = 'r')


# In[467]:


## we set the second step of Newton's method to zero to avoid outliers in the plot.
x_star_5_3a[3][1] = 0

f_x_difference_Newtons = x_star_5_3a[3]
f_x_difference_GD = x_star_5_3b[3]
f_x_difference_BTLS = x_star_5_3c[3]


# In[468]:


plt.plot(axis, f_x_difference_Newtons, color = 'b')
plt.plot(axis, f_x_difference_GD, color = 'g')
plt.plot(axis, f_x_difference_BTLS, color = 'r')


# In[ ]:




