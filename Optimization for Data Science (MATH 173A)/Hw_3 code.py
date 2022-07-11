#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
import random as rand


# In[2]:


training_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_train.csv")


# In[3]:


test_data = pd.read_csv("D:\\a_School Things\\Super Senior (Fall 2021)\\Fall 2021\\Math 173A\\hw 3\\mnist_test.csv")


# In[4]:


test_data


# In[5]:


## we extract the index values of each row, then define a new variable for each numerical value we wish to isolate.

indicies = training_data.index

zero_indicies = indicies[training_data['5'] == 0] 
ones_indicies = indicies[training_data['5'] == 1] 
two_indicies = indicies[training_data['5'] == 2] 
three_indicies = indicies[training_data['5'] == 3] 
four_indicies = indicies[training_data['5'] == 4] 
five_indicies = indicies[training_data['5'] == 5] 
six_indicies = indicies[training_data['5'] == 6] 
seven_indicies = indicies[training_data['5'] == 7] 
eight_indicies = indicies[training_data['5'] == 8] 
nine_indicies = indicies[training_data['5'] == 9] 


# In[7]:


## now we select a random index from each value and print it

rand_zero = rand.choice(zero_indicies.tolist())
print(rand_zero) 

## In the next line we isolate the random index, and convert all but the first column, which contains
## the numerical identifier, to a numpy array
choice_row_0 = training_data.loc[training_data.index == rand_zero].iloc[: , 1:].to_numpy()

## We print the reshaped image
plt.imshow(choice_row_0.reshape((28,28)), cmap = 'gray')

## we repeat the process for all nine digits


# In[8]:


rand_one = rand.choice(ones_indicies.tolist())
print(rand_one) 

choice_row_1 = training_data.loc[training_data.index == rand_one].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_1.reshape((28,28)), cmap = 'gray')


# In[9]:


rand_two = rand.choice(two_indicies.tolist())
print(rand_two) 

choice_row_2 = training_data.loc[training_data.index == rand_two].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_2.reshape((28,28)), cmap = 'gray')


# In[11]:


rand_three = rand.choice(three_indicies.tolist())
print(rand_three) 

choice_row_3 = training_data.loc[training_data.index == rand_three].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_3.reshape((28,28)), cmap = 'gray')


# In[12]:


rand_four = rand.choice(four_indicies.tolist())
print(rand_four) 

choice_row_4 = training_data.loc[training_data.index == rand_four].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_4.reshape((28,28)), cmap = 'gray')


# In[13]:


rand_five = rand.choice(five_indicies.tolist())
print(rand_five) 

choice_row_5 = training_data.loc[training_data.index == rand_five].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_5.reshape((28,28)), cmap = 'gray')


# In[15]:


rand_six = rand.choice(six_indicies.tolist())
print(rand_one) 

choice_row_6 = training_data.loc[training_data.index == rand_six].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_6.reshape((28,28)), cmap = 'gray')


# In[17]:


rand_seven = rand.choice(seven_indicies.tolist())
print(rand_one) 

choice_row_7 = training_data.loc[training_data.index == rand_seven].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_7.reshape((28,28)), cmap = 'gray')


# In[18]:


rand_eight = rand.choice(eight_indicies.tolist())
print(rand_eight) 

choice_row_8 = training_data.loc[training_data.index == rand_eight].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_8.reshape((28,28)), cmap = 'gray')


# In[21]:


rand_nine = rand.choice(nine_indicies.tolist())
print(rand_one) 

choice_row_9 = training_data.loc[training_data.index == rand_nine].iloc[: , 1:].to_numpy()
plt.imshow(choice_row_9.reshape((28,28)), cmap = 'gray')


# In[22]:


## The first column of the dataframe, which holds the number identities, is referred to as '7' in the test data, and as '5'
## in the training data

## the below first selects uses .loc[] to isolate all members of the '7' column with '1' as their value, then .iloc[] selects
## the first 500 of these values. .to_numpy() reassigns these columns to an array instead of a dataframe so they may be iterable

x_ones_te = ((test_data.loc[test_data['7'] == 1]).iloc[0:500]).to_numpy()
x_zeros_te = ((test_data.loc[test_data['7'] == 0]).iloc[0:500]).to_numpy()


# In[23]:


## now we stack the test data into one vector of the form:
## [500_ones, 500_zeros]

x_te_all = np.concatenate((x_ones_te, x_zeros_te))
len(x_te_all)


# In[24]:


## we do the same thing for the training data, which has a column labled '5' containing its indicies

x_ones_tr = ((training_data.loc[training_data['5'] == 1]).iloc[0:500]).to_numpy()
x_zeros_tr = ((training_data.loc[training_data['5'] == 0]).iloc[0:500]).to_numpy()


# In[25]:


## We stack the training data the same way

x_tr_all = np.concatenate((x_ones_tr, x_zeros_tr))
print(len(x_tr_all))

## here we notice that our vectors are one index larger than they ought to be for the calculation. This is the number index
print(len(x_tr_all[0]))


# In[26]:


## in the next two cells we delete the first rows of our training and test data which contain only the 1 or 0 that 
## indicates the identity of the numeral

x_tr_all = np.delete(x_tr_all, 0, axis = 1)
x_te_all = np.delete(x_te_all, 0, axis = 1)
print(len(x_tr_all[0]), len(x_te_all[44]))


# In[28]:


## now we need to label our x-vectors with appropriate y-values. I have stacked my x-vectors so that the first 500 are ones, 
## and the next 500 are zeros. To assign appropriate y-values, all we have to do is concatenate a 500-length vector of 1s
## with a 500 length vector of -1s so long as the concatenation order is aligned with that of the Xs.

y_tr_ones = np.ones(500)
y_tr_zeros = -1*np.ones(500)
y_tr_all = np.concatenate((y_tr_ones, y_tr_zeros))

## we can use this y-vector for both the training and test data, since they were constructed in the same way. 


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


## I was recieving a lot of Math Overflow errors, and I was uncertain whether they were coming from exponential terms that were
## too large, or too small (probably too large). This function breaks the calculation of an exponential dot-product by
## first calculating the individual terms of the product, then multiplying them together in sequence. 

def split_exp_calc (w_t, x, y):
    product = 1
    for i in range(0, len(w_t)):
        product = product * math.exp(w_t[i]*x[i]*y)
    
    return product


# In[30]:


def calc_gradient(w_t, x_vector, y_vector):
    
    v_size = len(x_vector)
    sum = np.zeros(len(w_t), dtype = np.float64)  

    for i in range(0, v_size):
        
        sum += x_vector[i] * (-1*y_vector[i])/(1 + split_exp_calc(w_t, x_vector[i], y_vector[i]))
        
    return (sum/v_size)


# In[35]:


def f_of_w (w_t, x, y):
    
    sum = 0
    for i in range(0, len(x)):
        sum = sum + math.log(1 + split_exp_calc(w_t, x[i], -y[i]))
        
    return sum/(len(x))


# In[63]:


def descend_3(w_t, x_vector, y_vector, mu, count, ceiling):
    
    ## here we calculate our initial gradient at input w_t, and instantiate a new array to add F(w) terms to with each GD step
    
    gradient = calc_gradient(w_t, x_vector, y_vector)
    F_w_record = np.empty(0)
    
    while count < ceiling:
        w_t -= mu*gradient
        gradient = calc_gradient(w_t, x_vector, y_vector)
        
        F_w_record = np.append(F_w_record, f_of_w(w_t, x_vector, y_vector))
        
        print(F_w_record[count])
        count = count + 1
        
        if la.norm(gradient) < 10**-3 :
            return w_t, F_w_record
        
        
    return w_t, F_w_record


# In[64]:


w_test2 = np.zeros(784)
x_star = descend_3(w_test2, x_tr_all, y_tr_all, 10**-6, 0, 200 )


# In[65]:


x_axis = np.array(range(0,len(x_star[1])))
plt.plot(x_axis, x_star[1])


# In[45]:


x_star[0]


# In[66]:


min = 1
min_index = 0

for i in range(1, len(x_star[1])):
    if x_star[1][i] < min:
        min = x_star[1][i]
        min_index = i
        
print(min, min_index)
x_star[1]


# In[67]:


## assign +1 or -1 to each of the test X values based on our idea w_t

y_vec_test = np.zeros(len(x_te_all))

for i in range (0, len(x_te_all)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(x_star[0], x_te_all[i]) < 0):
        y_vec_test[i] = -1
    else:
        y_vec_test[i] = 1
    


# In[68]:


## now we can compare the generated vector with the known identity vector y_tr_all
mistake_count = 0
for i in range (1, len(y_vec_test)):
    if(y_vec_test[i] != y_tr_all[i]):
        mistake_count += 1
        
print(mistake_count / len(y_vec_test))


# In[69]:


y_vec_test_tr = np.zeros(len(x_tr_all))

for i in range (0, len(x_tr_all)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(x_star[0], x_tr_all[i]) < 0):
        y_vec_test_tr[i] = -1
    else:
        y_vec_test_tr[i] = 1


# In[70]:


## now we can compare the generated vector with the known identity vector y_tr_all
mistake_count = 0
for i in range (1, len(y_vec_test)):
    if(y_vec_test[i] != y_tr_all[i]):
        mistake_count += 1
        
print(mistake_count / len(y_vec_test))


# In[46]:


x_nines_tr = ((training_data.loc[training_data['5'] == 9]).iloc[0:500]).to_numpy()
x_fours_tr = ((training_data.loc[training_data['5'] == 4]).iloc[0:500]).to_numpy()

x_nines_te = ((test_data.loc[test_data['7'] == 9]).iloc[0:500]).to_numpy()
x_fours_te = ((test_data.loc[test_data['7'] == 4]).iloc[0:500]).to_numpy()

x_te_all_nines_and_fours = np.concatenate((x_nines_te, x_fours_te))
x_tr_all_nines_and_fours = np.concatenate((x_nines_tr, x_fours_tr))


x_tr_all_nines_and_fours = np.delete(x_tr_all_nines_and_fours, 0, axis = 1)
x_te_all_nines_and_fours = np.delete(x_te_all_nines_and_fours, 0, axis = 1)

## we can use the same y vector since we have constructed our X the same as the 1s and 0s case


# In[47]:


print(len(x_tr_all_nines_and_fours[2]))
print(len(x_te_all_nines_and_fours[567]))


# In[55]:


w_test2_9_4 = np.zeros(784)
x_star_nines_and_fours = descend_3(w_test2_9_4, x_tr_all_nines_and_fours, y_tr_all, 10**-6, 0, 200 )


# In[59]:


x_axis = np.array(range(0,len(x_star_nines_and_fours[1])))
plt.plot(x_axis, x_star_nines_and_fours[1])


# In[60]:


y_vec_test_nines_and_fours = np.zeros(len(x_te_all_nines_and_fours))

for i in range (0, len(x_te_all_nines_and_fours)):
    
    ## if sign(w_t x_i < 0), x_i represents a zero figure, so we assign its index -1. 
    
    if(np.dot(x_star_nines_and_fours[0], x_te_all_nines_and_fours[i]) < 0):
        y_vec_test_nines_and_fours[i] = -1
    else:
        y_vec_test_nines_and_fours[i] = 1
    


# In[61]:


mistake_count_9_4 = 0
for i in range (1, len(y_vec_test_nines_and_fours)):
    if(y_vec_test_nines_and_fours[i] != y_tr_all[i]):
        mistake_count_9_4 += 1
        
print(mistake_count_9_4 / len(y_vec_test_nines_and_fours))


# In[ ]:




