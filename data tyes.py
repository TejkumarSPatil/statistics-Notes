# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:25:46 2019

@author: TEJ
"""

DATA TYPES: numerical, categorical and ordinal.
###################################################
NUMERICAL: continous and descete.

           CONTINOUS DATA:it is non-contable. (ex-rain falling 6.5mm were it is descrete)
                          data is normally distributed 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data=np.random.randint(1,500,100)
num_bins=5
plt.hist(data,num_bins)
plt.show()   

########################            
          DESCRETE DATA: it is countable.(ex- if roll the dies ie, six possibility 1 to 6)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x=[1,2,3,4]
y=[5,6,7,8]
plt.bar(x,y,label='modi')
plt.show()
          
##############################################################

CATEGORICAL DATA: which represents the characteristics of man ie. gender,nationality, marital status
                      
                      
       using pi plot
    
import matplotlib.pyplot as plt
 
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()      
                  
##################################################

ORDINAL DATA: it is a combination of numerical and categorical













