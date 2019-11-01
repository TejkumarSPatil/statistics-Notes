                                STATISTICS----------------------

- statistics is a branch branch of applied mathematics, were we collect the data, organise it and 
  do an interpretation on it and visualization on it   
- using this data for decision making.
- to get the overall description of data we use central tendency, measure of spread and
   Measures to describe shape of distribution


                                         CENTRAL TENDENCY
                                         
 CENTRAL TENDENCY- it gives distribution of data around the central value. it consist of three topics,
 
 > mean= total number divided by total count
 > median= we arrang the data either ascendind or descending order and take middle value gives as an median. 
 > mode = in given dataset frequently ocuuring element.
        
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline                      
            ## matplotlib inline used as visualization in their spyder

 ## lets create random number list ie,

m=np.random.randint(7,10,20)
print(m)

  # find mode, median and mean 

#  foe calculating mode
  
from statistics import mode
mode(m)

####   or used an array

n = np.array ([10,15,20,25,30,35,45])
print(n)

  #####     import numpy as np  
##### press    n
#####  create   array

######formula is            np.mean(n)
######                       np.std(n)     standard deviation
#########                  np.var(n)        variants
  #########             np.max(n)
  #######               np.min(n)

##########################################################################                 
 ###################################################################################                    

                               MEASURE OF SPREAD(dispertion)
                               
- As the measure of central value gives out the central value of the distribution,
  measures of dispersion describe the spread of data or the variation of data around the 
  central value.                               
- by using measure of spread we see population and sample varies as well.
- measure of spread is a variability in the data and how the data is distributed.                                
- it is used in election polls,judge your test cource, percentae increase in sallary
- it consist of 4 things, they are-- 
                                  range,quartile,varience and standard deviation
                                  
######                                  
   RANGE- it gives how well data is spread out & it is calculated as difference between highest and lower value in the dataset.                                

n=np.random.randn(4)
print(n)

np.max(n)-np.min(n)
######  or
n=np.random.randint(4,10,5)
print(n)

np.max(n)-np.min(n)

# in the above example we understood range but find range b/w age group 5 and 50 is difficult in this case we use quartile
 
#############

   QUARTILE- divided into four, they are median,25%,50% and 75% 
   
- makes it easy to work with data which is not symmetrically distributed and has outliers.  
- Mean, Median and mode is the numerical summary of the entire dataset which is symmetrically distributed whereas quartiles divide 
  our dataset into four equally sized groups based on five number summary:
            Minimum, first quartile, median, third quartile and maximum.
- The box in the box plot represents the 50 percent of the data values known as interquartile range (IQR).
- IQR indicates the variability in the set of values. 
- Large IQR means a large spread in values. 
- Small IQR indicates most of the values fall near the center of data. 
- Box plot shows minimum and maximum values through the whiskers which extends both the sides and also outlier points which extends beyond the whiskers.



n=np.random.randn(30)
print(n)                  # calculate median

# first quartile
q1=np.percentile(n,25)
print(q1)

# second quartile
q2=np.percentile(n,50)
print(q2)

# third quartile
q3=np.percentile(n,75)
print(q3)

# in the aove example second quartile is equal to the median

    INTERQUARTILE RANGE(IQR)
    
-  It’s the difference between the third quartile and the first quartile. 50% of the population data lies here.    
iqr=q3-q1
print(iqr) 

########################

    VARIANCE- how for the data is away(spread) from the mean and variance is changes for population and sample
              it is a rough idea of how data is away(spread) from the mean
              Variance is the average of all squared deviations.
              if more variance, better model to explain the data 
              
population=np.random.randn(100) 
print(population)
np.var(population) ##  var is submodule in numpy

sample=np.random.choice(population,30) 
print(sample)
np.var(sample)

################################

   STANDARD DEVIATION- it is a square root of variance
                       it gives exact value how data is away(spread) from the mean  
- Standard Deviation gives us an idea about the concentration of the data around the mean of the dataset.
- Standard deviation is low if the data is highly concentrated around the mean and vice versa.

np.std(population) 

np.std(sample)    

######################################################################

                          TYPES OF STATISTICS
                          
2 types: descriptive - description of data, we already known to us.ex-score of the cricket match.
         inferential - we try to infer the data which is unknown to us.ex-we know average height of the country, so we take perticular region average height. 

   DESCRIPTIVE STATISTICS-------
   
 basic example:
     
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from pandas.plotting import scatter_matrix
%matplotlib inline

df=pd.DataFrame(dict(id=range(6),age=np.random.randint(50,70,6)))
print(df) # in o/p, id is in proper order

df.age.mean() or # df['age'].mean()
df.describe()
df.id.mode() ## non of the element is frequently occured
df.age.max()-df.age.min()  # gives range
df.boxplot(column='age',return_type='axes')
######################################################

                     Measures to describe shape of distribution:
                         
                         
             skewness and kurtosis
 
 skewness= distribution differs from a normal distribution.
        The skewness is a parameter to measure the symmetry of a data set 
            describe the lack of symmentry from the mean, a perfectly symmetrical data have skewness zero
          ex:normal distrbution skewness has zero. 2 types
          negative skewness.
          positive skewness.
          
 kurtosis= kurtosis to measure how heavy its tails are compared to a normal distribution
           find the peak in probably distribution curve,then go head new measure of curtosis to find this.
           positive curtosis:which represents distribution is more peak towards the normal distribution.
           negative curtosis:which represents distribution is less peak towards the normal distribution.  

    
- These functions calculate moments of the probability density distribution (that's why it takes only one parameter) and
    doesn't care about the "functional form" of the values.
    
 If Mode< Median< Mean then the distribution is positively skewed.

If Mode> Median> Mean then the distribution is negatively skewed.

   
    
df['age'].skew() # positive skewed

df['age'].kurt() # negative curtosis

 ###########################   or  
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.DataFrame(dict(id=np.random.randint(15,40,6),age=np.random.randint(50,70,6)))
print(df)  # in o/p, id is not proper order    
     
##########################################################################

  INFERENTIAL STATISTICS---------------
  
- inferential statistics analysis infers properties of a population. 
        ex-hypothesis testing and deriving estimates
  
                     POPULATION AND SAMPLE
                     
- POPULATION IN STATISTICS MEANS TOTAL OBSERVATION THAT IS MADE.
- SAMPLE IS THE SUBSET OF POPULATION, SO THAT IT DESCIBE THE POPULATION IN SUCH A WAY WE WOULD INTERPRETE IT.                    
                     
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

population=np.random.randint(10,20,500) ## random is module under numpy & randint is the submodule
print(population)

np.mean(population)
np.median(population)
                    
from statistics import mode
mode(population)  

# next we take the one sample

sample=np.random.choice(population,20)  # were 20 is total num of population                  
print(sample)   ##  calculate mean median mode ie np.mean(sample) 

# we observe that sample mean,median and mode little closer to population mean,mode and median

  ##    next we take the multiple samples
sample1=np.random.choice(population,100)  
sample2=np.random.choice(population,100)
sample3=np.random.choice(population,100)
sample4=np.random.choice(population,100)

all_samples=[sample1,sample2,sample3,sample4]
sample_mean=[]

for i in all_samples:
    sample_mean.append(np.mean(i))

sample_mean  # we get each sample mean

np.mean(sample_mean) # we observe that all sample mean is little closer to population mean

pd.DataFrame(sample_mean).plot(kind='density')   ##  data is symmetical distributed

   POINT OF ESTIMATION
   
--CONFIDENCE INTERVAL= it usually covers 95%of data and it consist of lower and upper limit(95% data is available)
     
CONFIDENCE INTERVAL = (sample_mean - margin of error(lower limit) , sample_mean + margin of error(upper limit))             
                       margin of error formula is der in notes
                       
# import scipy.stats as stats
# z_critical=stats.norm.ppf(q=0.975)  # percent point function
# t_critical=stats.t.ppf(q=0.975,df=24) # df is the deree of freedom
                       

margin_of_error= z_critical * (np.std(sample_mean)/np.sqrt(100))

## lower limit
np.mean(sample_mean)-margin_of_error 

## upperlimit
np.mean(sample_mean)+margin_of_error                  

############################# #############################    
                        
#Create a Dictionary of series

import pandas as pd
import numpy as np

d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack',
   'Lee','David','Gasper','Betina','Andres']),
   'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
   'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])}

df = pd.DataFrame(d)
print(df)

##   df.sum()

    axis=1
##   df['Age']+df['Rating']  this is addition of age and ratings
    
    mean()
##  df.mean()  
    
    std()   ## standard deviation
##  df.std()
    
##  df.describe()
    
##  df.describe(include=['object'])
    
##  df. describe(include='all')    #####  ask sir
    
   Pandas – data handling
Numpy – statisticd and array handling

12/19/2018
#####    import os
#######   os.chdir('C:\\Users\\USER\\Downloads')
######     import pandas as pd
########## data=pd.read_csv('diabetes.csv')
#######   data

####  WE WANT ONLY SINGLE COLUMN
#####   data.columns
###### data2 = data [["BMI"]] 
#######   data2
#####  len(data)      
#####   len(data.columns)


###############################################

###############################   20/12/2018
import pandas as pd
import numpy as np
         import os
         os.chdir('C:\\Users\\USER\\Downloads')
         import pandas as pd
         data=pd.read_csv('diabetes.csv')
         data

######   data.describe()
######   data.columns.values
######   data[1:3]       finding rows
######   data['Alcohol'][0]   o/p is 14.23  very very imp


     data.mean()
     data.median()    how 2 get mean & median at a time because comparing  ask sir
     data.describe()
     
     
     
     
#   df[0:5]['salary']   ### for first 5 rows for the column named salary.

#   df.loc[[1,3,5],['salary','name']]    ##  Reading Specific Columns and Rows

#   df.loc[2:6,['salary','name']]   #### Reading Specific Columns for a Range of Rows





######   aaa=data.describe()
######   aaa.columns

######   aaa['BloodPressure']
######   aaa['BloodPressure','Glucose'] we want two columns not possible because string ie, possible only in integers
######   aaa['Glucose':'Insulin']       ask sir
######   aaa['Glucose']
######   aaa['DiabetesPedigreeFunction']

######   data['Age'].max()

######   bbb=data.groupby(['Outcome']).size()
######   bbb

######   bbb=data.groupby(['Outcome']).count()
######   bbb     

######   ccc=data.groupby(['Age']).size()
######   ccc
     
     df1.groupby(['Gender','Property_Area']).size().reset_index(name='counts')   
     # above command gives the how many males in rural and how many females in urban
     sns.countplot(x='Gender',hue='Property_Area',data=df1)
     
         df.groupby(['Gender']).size().plot(kind='bar') ## using bar plot in test excell
         df.groupby(['Gender']).size().plot(kind='pie') ## using pi chart
         df.groupby(['Gender']).size().plot(kind='hist') # using histogram
         df.groupby(['Gender']).size().plot(kind='box')  # using box plot
         
         import seaborn as sns
         %matplotlib inline
         
         sns.barplot(x='Gender',y='Credit_History',data=df)  # here gender is a string and credit_history is a integer
         sns.boxplot(x='Gender',y='Loan_Amount_Term',data=df) # boxplot for gender and loan amount
         sns.boxplot(data=df) # boxplot for all columns in the test excell
         sns.boxplot(data=df,orient='h') # for horizontal view
         sns.stripplot(x='Gender',y='Loan_Amount_Term',data=nan1)# data is overlapped so we use swarmplot
         sns.swarmplot(x='Gender',y='Loan_Amount_Term',data=nan1) # here data is not overlapped

######   max(data['Glucose'])   or data['Glucose'].max()   maximum glucose level
######   min(data['Glucose'])      minimum glucose level
######   max(data['Pregnancies'])  maximum num of pragnecy

######   data['Age'].mean()        mean of age
######   data['BMI'].mean()        mean of age

######    The Age below 50 and above 50 in DiabetesPedigreeFunction
######    ddd=data.loc[data['Age']>=50]  ###  ask sir i want total number
     
######    eee=data.loc[data['Age']<=50]

######           ####  it gives single row information
######    data.iloc[5]  it gives single row information

######    pd.set_option('display.max_columns',12)
######    data

######    pd.set_option('display.max_rows',12)
######    data

######    pd.set_option('display.max_columns',12)
######    data

######    data.loc[data['Age']==81]  perticular information Age 81

######    data.loc[data['Age']>=81]

#######################################
    HISTOGRAM
 > A histogram is an accurate graphical representation of the distribution of numerical data.   
 > It is an estimate of the probability distribution of a continuous variable (quantitative variable) and was first introduced by Karl Pearson.   
 > It is a kind of bar graph.
 > To construct a histogram, the first step is to “bin” the range of values — that is,
   divide the entire range of values into a series of intervals — and then count how many values fall into each interval.
 > The bins are usually specified as consecutive, non-overlapping intervals of a variable.
 > The bins (intervals) must be adjacent, and are often (but are not required to be) of equal size.
 > Basically, histograms are used to represent data given in form of some groups. 
 > X-axis is about bin ranges where Y-axis talks about frequency.
 > So, if you want to represent age wise population in form of graph then,
   histogram suits well as it tells you how many exists in certain group range or bin, if you talk in context of histograms.



    2/1/2018
######     describe function in pandas assignment descriptive statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
                 #%matplotlib inline

## df = pd.read_csv("E:\\python\\train.csv") #Reading the dataset in a dataframe using Pandas
## df
                 
##  df.head(10) or df[0:10]  first ten rows of data from excel  
                 
##  df.describe()  if it is not come use pd.set_option('display.max_columns',13) after press df        
   
## df['Property_Area'].value_counts()  or groupby option  total num of urban,rural and semiurban            
                 
## df['Property_Area']
                 
##  df['ApplicantIncome'].hist(bins=50)    graph came
                 
##  df.boxplot(column='ApplicantIncome')      
                 
##  df.boxplot(column='ApplicantIncome', by = 'Education')
                 
##  df['LoanAmount'].hist(bins=50)
                 
 here includes some extreme values in loan so we need to do data munging
data munging:
  ### larger value compress to the smaller so use log function
  
##   df.apply(lambda x: sum(x.isnull()),axis=0)          

##   df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)    press df
    
##  df['LoanAmount_log'] = np.log(df['LoanAmount']) press df
    
##  df['LoanAmount_log'].hist(bins=20)

##   df['LoanAmount_log'] = np.log(df['LoanAmount'])
##   df['LoanAmount_log'].hist(bins=20)

##   df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    
##   df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    
##   df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    
##    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    
##     df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)  
    
    #####################################################
## line chart 
from pylab import *
t = arange(0.0, 2.0, 0.01)
s = sin(2.5*pi*t)
plt.plot(t, s,label='sine wave')

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('Sine Wave')
grid(True)
show()


### histogram
##  it is continous,values should be attached

## A histogram shows the frequency on the 
##vertical axis and the horizontal axis is another dimension. 
##Usually it has bins, where every bin has a minimum and maximum value.
##Each bin also has a frequency between x and infinite.

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
 
x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.8)
plt.show()


#####bar plots

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
 
plt.show()


####  ##############     pie chart

import matplotlib.pyplot as plt
 
# Data to plot
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()


####scatter plot
##  where is the dense----
###A scatter plot is a type of plot that 
##shows the data as a collection of points. 
#The position of a point depends on its two-dimensional value,
## where each value is a position on either
##the horizontal or vertical dimension.

import numpy as np
import matplotlib.pyplot as plt
 
# Create data
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3
 
# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot vtricks.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

### box plot

import matplotlib.pyplot as plt
 
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52,500]
####   dvalue1=pd.DataFrame(value1)   converting list to dataframe, calculating mean
###  dvalue1.describe()
###  dvalue1.mean() 
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
 
box_plot_data=[value1,value2,value3,value4]
plt.boxplot(box_plot_data)
plt.show()

## define x axis and y axis

#################################################################################


                             MATPLOTLIB (DATA VISUALIZATION)
                             
> Matplotlib is a python library used to create 2D graphs and plots by using python scripts. 
> It has a module named pyplot which makes things easy for plotting by providing feature to control line styles, font properties, formatting axes etc. 
> It supports a very wide variety of graphs and plots namely - histogram, bar charts, power spectra, error charts etc.


############################################################
#####  HOMEWORK
###  GRAPH BASIC EXAMPLES

                       sine wave-
                       
from pylab import *                       
import numpy as np                       
import pandas as pd
import matplotlib.pyplot as plt
t = arange(0.0, 2.0, 0.01)
s = sin(2.5*pi*t)
plot(t, s)

xlabel('time')
ylabel('voltage')
title('Sine Wave')
grid(True)
show()
#######################################################
                      COS wave-
from pylab import *                       
import numpy as np                       
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
t = arange(0.0, 2.0, 0.01)
s = cos(2.5*pi*t)
plot(t, s)

xlabel('time')
ylabel('voltage')
title('cos Wave')
grid(True)
show()
############################################################
                      tan wave-
from pylab import *                       
import numpy as np                       
import pandas as pd                      
import matplotlib.pyplot as plt
t = arange(0.0, 2.0, 0.01)
s = tan(2.5*pi*t)
plot(t, s)

xlabel('time')
ylabel('voltage')
title('Tan wave')
grid(True)
show()                      
###################################################

                          PLOTS
import numpy as np                       
import pandas as pd                          
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[5,6,7,8]
plt.plot(x,y)
plt.legend()
plt.show()


import matplotlib.pyplot as plt
x=[1,2,3,4,19]
y=[5,13,8,11,19]
plt.plot(x,y)
plt.legend()
plt.show()

####  giving x-axis and y-axis
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[5,6,7,8]
plt.plot(x,y,label='army')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Indian')
plt.legend()
plt.show()
###################################
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[5,6,7,8]

x1=[10,20,30,40]
y1=[100,90,80,70]

plt.plot(x,y,label='Namo')   ###  ask sir
plt.plot(x1,y1,label='Modi') ###  here we put the size option but it not came

plt.xlabel('Temp')
plt.ylabel('y')
plt.title('Indian')
plt.legend()
plt.show()
###################################
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[5,6,7,8]

x1=[10,20,30,40]
y1=[100,90,80,70]

plt.plot(x,y,label='Namo',color='c')   ###  ask sir
plt.plot(x1,y1,label='Modi') ###  here we put the size option but it not came

plt.xlabel('Temp')
plt.ylabel('y')
plt.title('Indian')
plt.legend()
plt.show()

##################################################################
                      BAR-PLOTS
                      
     it showa a relation between numerical and categorical variable
      ex- individual scores of each player.                 

import matplotlib.pyplot as plt
x=[2,4,6,8,10]  
y=[6,7,8,2,4]   # were x and y are arbitary bars
plt.bar(x,y,label='Bars1')  ##  we getting single bar
##############

import matplotlib.pyplot as plt
x=[2,4,6,8,10]  
y=[6,7,8,2,4]   
plt.barh(x,y,label='Bars1') # it shows a horizental passion

##############
import matplotlib.pyplot as plt
x=[20,40,60,80,100]  
y=['aa','bb','cc','dd','ee']   # were x and y are arbitary bars
plt.bar(x,y,label='Bars1')
###############
              
import matplotlib.pyplot as plt
x=[2,4,6,8,10]
y=[6,7,8,2,4]
plt.bar(x,y,label='Indian')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()

################################################

import matplotlib.pyplot as plt
x=['aa','bb','cc','dd','ee','ff']
y=[0,20,40,60,80,100]
plt.bar(x,y,label='Bars1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('teja')
plt.legend()
plt.show()
#############################################

import matplotlib.pyplot as plt
x=[2,4,6,8,10]
y=[6,7,8,2,4]

x1=[1,3,5,9,11]
y1=[3,4,2,8,7]

plt.bar(x,y,label='Bars1')
plt.bar(x1,y1,label='Bars2')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()

#################################################

import matplotlib.pyplot as plt
x=[2,4,6,8,10]
y=[6,7,8,2,4]

x1=[1,3,5,9,11]
y1=[3,4,2,8,7]

plt.bar(x,y,label='Bars1',color='c')  ### c=cyan(c0lor name)
plt.bar(x1,y1,label='Bars2',color='r')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()
#####################################################
                      
                        HISTOGRAM -
            it is continous,values should be attached

import matplotlib.pyplot as plt
x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
num_bins = 5
plt.hist(x, num_bins, facecolor='blue', alpha=0.8)
plt.show()

###########################################
import matplotlib.pyplot as plt
x=[12,23,34,45,56,67,78,89,90,9,98,7,8,9,65,43,21,23,43,13,69,81,82,85,86,87]
num_bins=5
plt.hist(x,num_bins,label='Indian',color='r',alpha=0.20)

plt.xlabel('Namo') ###  ask sir (label name not comming)
plt.ylabel('Modi')
plt.title('BJP')
plt.show()

########################################

                      SCATTERED PLOTS
      it is a two dimensional data visualization
      one variable plotted in x-axis and another variable plotted in y-axis                
      it is used find relationship between two variables. 

stripplot= one dimensional data visualization. were one variable is categorical.this is used when sample size is small.
           it is something similar to the scattered plot               
    
import matplotlib.pyplot as plt
x=[2,3,4,5,6]
y=[9,8,6,5,4]
plt.scatter(x,y,label='teju',color='r')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()

##########################################

import matplotlib.pyplot as plt
x=[2,3,4,5,6]
y=[9,8,6,5,4]
plt.scatter(x,y,label='teju',color='r',marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()
###################################################

import numpy as np
import matplotlib.pyplot as plt
 
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
plt.scatter(x,y,label='teju',color='b',marker='*')

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()

##############################################

import matplotlib.pyplot as plt
x=[2,3,4,5,6]
y=[9,8,6,5,4]
plt.scatter(x,y,label='teju',color='r',marker='2',s=100) ##goto google type matplotlib marker option

plt.xlabel('x')
plt.ylabel('y')
plt.title('teju')
plt.legend()
plt.show()

##################################################
                      
                    PI CHART
   it divides circle into multiple slices                 
                    
                    
import matplotlib.pyplot as plt
 
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()  
#######################################

import matplotlib.pyplot as plt
 
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 1, 0, 0)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()
#############################################

import matplotlib.pyplot as plt
 
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0.1, 0)  
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
 
plt.axis('equal')
plt.show()
#############################################
import matplotlib.pyplot as plt
 
labels = ['Python', 'C++', 'Ruby', 'Java']
sizes = [215, 300, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0, 0.1)
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
 ## autopct: None (default), string, or function, optional
#       If not None, is a string or function used to label the wedges with their numeric value. 
#       The label will be placed inside the wedge. If it is a format string, the label will be fmt%pct. 
#       If it is a function, it will be called.
 #      pctdistance : float, optional, default: 0.6
plt.axis('equal')
plt.show()                  

##########################################################
                               
                               BOX PLOT
> Boxplots are a measure of how well distributed the data in a data set is.
> It divides the data set into three quartiles.
> This graph represents the minimum, maximum, median, first quartile and third quartile in the data set. 
> It is also useful in comparing the distribution of data across data sets by drawing boxplots for each of them.                           


import matplotlib.pyplot as plt 
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
####   dvalue1=pd.DataFrame(value1)   converting list to dataframe, calculating mean
###  dvalue1.describe()
###  dvalue1.mean() 
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
 
plt.boxplot([value1,value2,value3,value4])
plt.show()
##########################################
import matplotlib.pyplot as plt
 
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
 
plt.boxplot([value1,value2,value3,value4])

plt.xlabel('Namo')
plt.ylabel('Modi')
plt.title('BJP')
plt.show()

#############################
import matplotlib.pyplot as plt
 
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52,25]
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
 
plt.boxplot([value1,value2,value3,value4])

plt.xlabel('Namo')
plt.ylabel('Modi')
plt.title('BJP')
plt.show()
          ## in the output there  is no outlayers because 25 is added to the value1

##################################################
          
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.boxplot(grid='True')
############################################################

 
import matplotlib.pyplot as plt
 
value1 = [1,51,52,53,54,55,56,57,58,59,60,61,62,75,76]
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
 
plt.boxplot([value1,value2,value3,value4])

plt.xlabel('Namo')
plt.ylabel('Modi')
plt.title('BJP')
plt.show()

################################################

                      IRIS DATASETS
                      
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from pandas.plotting import scatter_matrix

iris = sns.load_dataset("iris")
print(iris.head())
    
print (iris.shape)

iris.target

print(iris.describe())

iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False,sharey=False,)
iris.plot(kind='box', subplots=True, layout=(4,4), sharex=False,sharey=False,)
iris.hist()
scatter_matrix(iris) # it shows the high correlation and predictable relationship
plot.show()


x=iris.iloc[:,0:3].values
y=iris.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3,random_state=0) 
  
apply all algorithms to check accuracy

################################################### or
from sklearn import datasets
iris=datasets.load_iris()
iris
print(iris.data.shape)
print(iris.target_names)
#####################################

  apply KMeans algorithm on iris datasets

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans  ####  run for algorithm

iris=datasets.load_iris()
print(iris)
there is 150 observations and 4 feactures

x=pd.DataFrame(iris.data)

x.head()

x.columns=['sepal_length','sepal_width','petal_length','petal_width']
x.head()
 ######################### now apply the algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ##  for scattered plot
%matplotlib inline
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.cluster import KMeans  ####  run for algorithm
        

iris=datasets.load_iris()
print(iris)

x=pd.DataFrame(iris.data)

x.head()

x.columns=['sepal_length','sepal_width','petal_length','petal_width']
x.head()

a=KMeans(n_clusters=3)
a.fit(x)

a.labels_

colormap=np.array(['Red','Blue','Green'])
z=plt.scatter(x.sepal_length,x.sepal_width,x.petal_length,c=colormap[a.labels_])

accuracy_score(iris.target,a.labels_)

#######################################################################


ERROR


from sklearn import svm
from sklearn import datasets

iris=datasets.load_iris()

type(iris)

iris.data

iris.shape

iris.feature_names # in feature c is not der

iris.target

iris.target_names

x=iris.data[:,2] # which takes only 2 column
y=iris.target
x
y

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=4)
 
model=svm.SVC(kernel='linear')

model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
accuracy=model.score(x_test,y_test)
print(accuracy)











