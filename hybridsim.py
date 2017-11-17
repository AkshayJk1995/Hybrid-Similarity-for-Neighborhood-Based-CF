# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:51:24 2016

@author: AkshayJk
"""
import numpy
import csv
import math
import statistics
import time
from numba import jit
#import random
start_time = time.time()
check_time=time.time()
no_of_users=100
no_of_items=100
a=numpy.zeros((100,100)) #rows -users, columns- items, elements- ratings
r=numpy.zeros((no_of_users,no_of_items)); #stores predictive ratings for each a[i][j]
max=10
rmed=5
f=open('samp_bx1.csv','r')
reader=csv.reader(f)
count=0
for row in reader:
    a[count]=[int(i) for i in row]
    count+=1
    
@jit(nopython=True)    
def find_no_of_n(n,x): #finds no of occurences of x in n
    count=0
    for i in range(len(n)):
        if n[i]==x:
            count+=1
    #print("in fond_no_of_n")
    return count

@jit      
def find_nonzero(n): #finds no of non zero elements in vector n
    count=0
    for i in range(len(n)):
        if n[i]!=0:
            count+=1
    #print("in find_nonzero")
    return count

@jit
def bhattacharya(a,b):
    n1=find_nonzero(a)
    n2=find_nonzero(b)
    res=0
    #print("in Bhattacharya")
    for i in range(1,6):
        prod1=float(find_no_of_n(a,i))/float(n1)
        prod2=float(find_no_of_n(b,i))/float(n2)
        res+=math.sqrt(prod1*prod2)
    return res
    
@jit
def jaccard_similarity1(a,b):
    count1=0
    count2=0
    i=0
    #print("in Jaccard")
    for i in range(len(a)):
        if a[i]!=0 and b[i]!=0:
            count1+=1                        
        if a[i]!=0 or b[i]!=0:
            count2+=1
    if count2==0:
        return 0
    i+=1
    while i<len(b):
        if b[i]!=0:
            count2+=1
        i+=1
    while i<len(a):
        if a[i]!=0:
            count2+=1
        i+=1
    return float(count1)/float(count2)

@jit
def loc(a,b,i,j):  
    avg_a=numpy.sum(a)/float(len(a))
    avg_b=numpy.sum(b)/float(len(b))
    std_a=statistics.stdev(a)
    std_b=statistics.stdev(b)
    #print("In local")
    #loc_sim=(((float(a[0])-avg_a)*(float(b[0])-avg_b))+((float(a[1])-avg_a)*(float(b[1])-avg_b))+((float(a[2])-avg_a)*(float(b[2])-avg_b))+((float(a[3])-avg_a)*(float(b[3])-avg_b))+((float(a[4])-avg_a)*(float(b[4])-avg_b))+((float(a[5])-avg_a)*(float(b[5])-avg_b)))/float(std_a*std_b)
    loc_sim=((a[i]-avg_a)*(b[j]-avg_b))/(std_a*std_b)
    return loc_sim

@jit    
def new_sim(x,y):
    bhatt_sum=0.0
    #print("in new_sim")
    #complete bhattacharya with jaccard
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i]!=0 and y[j]!=0 and i!=j:
                bhatt_sum+=bhattacharya(a[:,i],a[:,j])*(pip_sim(x,y))
                #print("Local",loc(x,y,i,j))
                #print("bhatt",bhattacharya(a[:,i],a[:,j]))
                
    new_measure=jaccard_similarity1(x,y)+bhatt_sum
    return new_measure

@jit
def pearson(x,y):
    s=0
    t1=0
    t2=0
    avg1=sum(x)/find_nonzero(x)
    avg2=sum(y)/find_nonzero(y)
    for i in range(len(x)):
        if x[i]!=0 and y[i]!=0:
            s+=(x[i]-avg1)*(y[i]*avg2)
            t1+=pow((x[i]-avg1),2)
            t2+=pow((y[i]-avg2),2)
    if t1!=0 and t2!=0:
        res=s/(math.sqrt(t1)*math.sqrt(t2))
        return res
    return 0
        
@jit
def mmd(a,b):
    r=0
    t1=numpy.zeros(5)
    t2=numpy.zeros(5)
    i1=find_nonzero(a)
    i2=find_nonzero(b)
    for i in range(5):
        t1[i]=find_no_of_n(a,i+1)
        t2[i]=find_no_of_n(b,i+1)
    #print(t1)
    #print(t2)
    s=0.0
    for i in range(len(a)):
        if a[i]!=0 and b[i]!=0:
            x=t1[a[i]-1]
            y=t2[b[i]-1]
            s+=pow((x-y),2)
            r+=1
    if r!=0:
        sfin=s/r
        sfin=sfin-(1/i1)-(1/i2)
        sfin+=1
        ans=1/sfin
        return ans
    else:
        return 0
        
	
def cosine(a,b):
    prod_sum=0
    a2=0
    b2=0
    for i in range(len(a)):
        if a[i]!=0 and b[i]!=0:
            prod_sum+=a[i]*b[i]
            a2+=a[i]*a[i]
            b2+=b[i]*b[i]
    if a2!=0 and b2!=0:
        simi=float(prod_sum)/float(math.sqrt(a2)*math.sqrt(b2))
    else:
        return 0
    return simi

    
@jit    
def k_closest(user): #finds k closest of a user
    values=numpy.zeros((no_of_users-1,2))
    count=0
    #print("in k_closest")
    for other_user in range(no_of_users):
        if other_user!=user:
            values[count][0]=new_sim(a[user],a[other_user])
            values[count][1]=other_user
            count+=1
    print("Values",values)
    print("value[0]",values[0][0])
    scores=numpy.zeros((max,2))
    for i in range(no_of_users-2):
        maxi=i
        for j in range((i+1),no_of_users-1):
            if values[j][0]>values[maxi][0]:
                maxi=j
        if maxi!=i:
            values[maxi],values[i]=values[i],values[maxi].copy()
    for i in range(max):
            scores[i]=values[i]
    return scores        
     
@jit
def agreement(r1,r2):
    if (r1>rmed and r2<rmed) or (r1<rmed and r2>rmed):
        return 1
    return 0

@jit    
def pip(r1,r2,k):
    d=0
    c=(abs(r1-2.5)+1)*(abs(r2-2.5)+1)
    if agreement(r1,r2)!=0:
       d=abs(r1-r2)
       imp=c
    else:
        d=2*abs(r1-r2)
        imp=1/c
    prox=pow((11-d),2)
    s=0
    for i in range(no_of_users):
        s+=a[i][k]
    avg=s/find_nonzero(a[:,k])
    if (r1>avg and r2>avg) or (r1<avg and r2<avg):
        popul=1+pow((((r1+r2)/2)-avg),2)
    else:
        popul=1
    return float(prox*imp*popul)
        
@jit    
def pip_sim(x,y):
    sim=0.0
    for i in range(len(x)):
        sim+=pip(x[i],y[i],i)
    return sim

@jit
def msd(a,b):
    s=0.0
    count1=0.0
    for i in range(len(a)):
        if a[i]!=0 and b[i]!=0:
            count1+=1
            s+=pow((a[i]-b[i]),2)
    if count1!=0.0:
        ms=s/count1
    else:
        ms=0
    return (1-ms)                        
    
        
@jit  
def user_predict(i): #computes predictive rating r[i][j] for each a[i][j]
    i_avg=numpy.sum(a[i])/len(a[i])
    den_sum=0.0 #denominator sum
    #print("in user_predict")
    scores=k_closest(i)
    #scores=scores_test[0:max]
    scores_avg=numpy.zeros((max))
    #array to store average ratings of each kth neighbour 
    #temp=numpy.array(max)
    temp=numpy.zeros(max)
    count=0
    for s in scores:
        temp[count]=s[1]
        count+=1
    for m in range(0,max): 
        t=int(temp[m])
        scores_avg[m]=float(numpy.sum(a[t]))/float(len(a[t]))
    for m in range(0,max):
        den_sum=den_sum+abs(new_sim(a[i],a[int(temp[m])]))
    for c in range(no_of_items):
        num_sum=0
        for b in range(0,max):
            num_sum+=new_sim(a[i],a[temp[b]])*(a[temp[b]][c]-scores_avg[b])
        r[i][c]=i_avg+(num_sum/den_sum)

        
@jit        
def rmse():
    rmse_sum=0
    for i in range(no_of_users):
        for j in range(no_of_items):
            rmse_sum+=math.pow(a[i][j]-r[i][j],2)
    rmse_val=float(rmse_sum)/float(no_of_users*no_of_items)
    return math.sqrt(rmse_val)
    
@jit        
def mae():
    mae_sum=0.0
    for i in range(no_of_users):
        for j in range(no_of_items):
            mae_sum+=math.fabs(a[i][j]-r[i][j])
        #print(mae_sum)
    mae_val=mae_sum/float(no_of_users*no_of_items)
    return mae_val

@jit
def find_above_4():
    count=0
    for i in range(no_of_users):
        for j in range(no_of_items):
            if a[i][j]>=4:
                count+=1
    above_4=numpy.zeros((count,2))
    count=0            
    for i in range(no_of_users):
        for j in range(no_of_items):
            if a[i][j]>=4:
                above_4[count]=(i,j)
                count+=1
    return above_4

print(a)
print(a.shape)

for i in range(no_of_users):
    user_predict(i)
    print(r)
    iter_time=time.time()-check_time
    print("----------------------Iteration ",i+1," done in ",iter_time,"----------------------------------------")
    check_time=time.time()
print(r)
print("RMSE: ",rmse())
print("MAE: ",mae())
#print("Above 4: ",find_above_4())

print("end")
#numpy.savetxt("output_100_bx1.csv",r,delimiter=",")
print("--- %s seconds ---" % (time.time() - start_time))
