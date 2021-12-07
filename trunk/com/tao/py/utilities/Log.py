'''
Created on Dec 7, 2021

@author: Shufang
'''
from datetime import datetime

filters=[]


def addFilter(filter):
    filters.append(filter)

def out(info):
    printOut=False
    for ifilter in filters:
        if ifilter in info:
            printOut=True
            break
    
    if len(filters)==0 or printOut:
    
        print(info)


def p(tag,itype,info): 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    out("{}--{}--{}--{}".format(dt_string,tag,itype,info))  
def sp(tag,itype,info,time): 
    out("{:.2f}--{}--{}--{}".format(time,tag,itype,info))      
      
def i(tag,info):
    p(tag,"INFO",info)
def si(tag,info,time):
    sp(tag,"INFO",info,time)

def d(tag,info):
    p(tag,"DEBUG",info)
def sd(tag,info,time):
    sp(tag,"DEBUG",info,time) 
    
def w(tag,info):
    p(tag,"WARNING",info)
def sw(tag,info,time):
    sp(tag,"WARNING",info,time) 
    
def e(tag,info):
    p(tag,"ERROR",info)
def se(tag,info,time):
    sp(tag,"ERROR",info,time) 

    