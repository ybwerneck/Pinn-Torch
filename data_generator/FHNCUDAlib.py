###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import csv

class FHNCUDA:
    parametersN=[""]

    @staticmethod
    def readout():
        read_csv_matrix = lambda file_path: [list(map(float, row)) for row in csv.reader(open(file_path, 'r'))]

        with open('./outputs/u.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            row_count = sum(1 for row in csvreader)
        with open('./outputs/u.csv', 'r') as csvfile:

            csvreader = csv.reader(csvfile)
            column_count = [len(row) for row in csvreader][0]

        print("Number of rows in the CSV file: ", row_count, column_count)
        
        return (read_csv_matrix('./outputs/u.csv')) , (read_csv_matrix('./outputs/v.csv')) ,(read_csv_matrix('./outputs/t.csv')),(read_csv_matrix('./outputs/p.csv')) 


    @staticmethod
    def run(x0,T,dt,rate,P="",use_gpu=False, regen=True,name="out.txt"):  
        with open('./u.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(x0)
    
        FHNCUDA.callCppmodel(T,dt,rate)
        return FHNCUDA.readout()

    
    @staticmethod
    def callCppmodel(T,dt,rate):  
     #   print("Calling solver")
        name="./a.out"
        args=name + " " + str(T) + " "+ str(dt)  + " " +str(rate)    
        
        print("kernel call:",args)
        output = subprocess.Popen(args,stdout=subprocess.PIPE,shell=True)
        string = output.stdout.read().decode("utf-8")
        print(len(string))

        