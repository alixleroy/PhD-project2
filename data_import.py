## Import data from solver 
import numpy as np 
import csv
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
def read_info(namefile):
    ## Read information for QA 
    res_qa=np.array([])
    with open(namefile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            res_qa=np.append(res_qa,float(row[0]))
    res_qa = np.reshape(res_qa, (res_qa.shape[0],1)) # reshape into a vector mx1
    return(res_qa)


