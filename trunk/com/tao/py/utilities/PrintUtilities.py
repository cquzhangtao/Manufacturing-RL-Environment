'''
Created on Apr 4, 2022

@author: cquzh
'''

import sys
import time

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

class Printer(object):
 
    def enable(self):
        self.file=open('log'+str(time.time()*1000.0)+'.txt', 'w')
        self.original = sys.stdout
        sys.stdout = Tee(sys.stdout, self.file)
    
    
    def disable(self):
        sys.stdout = self.original
        self.file.close()