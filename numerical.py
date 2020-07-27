# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:56:53 2020

@author: AllenPC
"""
import numpy as np

def num_range(n):
    if n==0:
        num=0
    elif n>0 and n<=0.3:
        num = 0.2
        return num
    elif n>0.3 and n<=0.6:
        num = 0.5
        return num
    elif n>0.6 and n<=31:
        num = np.round(n)
    elif n>31 and n<=50:
        num = 40
    elif n>50 and n<=75:
        num = 60
    elif n>75 and n<=100:
        num = 85
    elif n>100 and n<=150:
        num = 125
    elif n>150 and n<=600:
        num = 300
    elif n>600 and n<=1100:
        num = 1000
    elif n>1100 and n<1980:
        num = 1500
    elif n>=1980 and n<1990:
        num = 1985
    elif n>=1990 and n<2000:
        num = 1995
    elif n>=2000 and n<=2025:
        num = np.round(n)
    elif n>2025 and n<=3000:
        num = 2500
    elif n>3000 and n<=6000:
        num = 5000
    elif n>6000 and n<=20000:
        num = 12000
    elif n>20000 and n<=60000:
        num = 40000
    elif n>60000 and n<=110000:
        num = 100000
    elif n>110000 and n<=600000:
        num = 400000
    elif n>600000 and n<=1100000:
        num = 800000
    elif n>1100000 and n<=6000000:
        num = 4000000
    elif n>6000000 and n<=10000000:
        num = 7500000
    elif n>10000000 and n<=13000000:
        num = 11000000
    elif n>13000000:    
        num = 15000000
    return int(num)

num = num_range(0.55)



