import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

#%%
# 1

data = pd.read_csv('/Users/sydlo/.spyder-py3/titanic_train.csv')

# Summary Statistics of Numerical Data
print(data.describe())

# Histograms or Box Plots of Numerical Data
surv = data['Survived']
# Pulled next two lines from Google Search. I imagine it pulled it from the Pandas page.
int_to_text = {1: 'yes', 0: 'no'}
surv = surv.map(int_to_text)

plt.figure()
plt.hist(surv)
plt.xlabel('Survived')
plt.ylabel('Number of Passengers')

# The mojorit of Passengers Died on the Titanic with over 500. Roughly 350 Survived.

pclass = data['Pclass']
int_to_text_2 = {1: 'One', 2: 'Two', 3: 'Three'}
pclass = pclass.map(int_to_text_2)
plt.figure()
plt.hist(pclass)
plt.xlabel('Pclass')
plt.ylabel('Number of Passengers')
plt.title('Passengers in Each Class')

# The majority of Passengers were 3rd class (the lowest). There were more 1st class passengers 
# than 2nd class, although they amounts were close.


age = data['Age']
plt.figure()
plt.hist(age,bins = 8,rwidth = 0.8)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

# The majority of Passengers were between the ages of 20-40. With 20-30 year old passengers
# being the largest group.



# Pie Chart
S = 0
C = 0
Q = 0

embark = data['Embarked']
for i in embark:
    if i == 'S':
        S += 1
    if i == 'C':
        C += 1
    if i == 'Q':
        Q += 1

embark_counts = [S,C,Q]
label_embark = 'Southampton', 'Cherbourg', 'Queenstown'
plt.pie(embark_counts,labels=label_embark, autopct='%1.1f%%')
plt.title('Percentage of Passengers that Embarked from Each Port')

plt.show()

# Nearly 3/4 of the passengers got on the boat in Southampton. Roughly 19% and 9% got on
# at Cherbourg and Queenstown respectively.

# Correlation


pclass1 = data[data['Pclass']==1]
pclass2 = data[data['Pclass']==2]
pclass3 = data[data['Pclass']==3]

pclass1_counts = pclass1["Survived"].value_counts()
pclass2_counts = pclass2["Survived"].value_counts()
pclass3_counts = pclass3["Survived"].value_counts()


plt.bar([4,0], pclass1_counts)
plt.bar([1,5],pclass2_counts)
plt.bar([2,6],pclass3_counts)
plt.xticks([1,5],['Dead','Survived'])
corr_labels = '1st','2nd','3rd'
plt.legend(corr_labels)
plt.title('Amount of Passengers That Survived Based on Class')

print('Percentage of 1st class that survived: ',pclass1_counts[1]/(pclass1_counts[0]+pclass1_counts[1])*100)
print('Percentage of 2nd class that survived: ',pclass2_counts[1]/(pclass1_counts[0]+pclass1_counts[1])*100)
print('Percentage of 3rd class that survived: ',pclass3_counts[1]/(pclass1_counts[0]+pclass1_counts[1])*100)

# The graph and percentages of survived passengers show that being 1st class allowed for more
# of a chance to make it off the boat and be rescued
# Percentages can be skewed easily here as there are far more 3rd class passengers than 1st class
# However, this can be disregarded as more 1st class passengers were saved than 3rd class
# even though the total amount of 1st class passengers is much lower than 3rd class.

# Chi2

sex_embark = pd.crosstab(data['Embarked'],data['Sex'])
print(sex_embark)


c, p, dof, expected = scipy.stats.chi2_contingency(sex_embark)
print(p)
print(sex_embark-expected)

Pclass = data['Pclass']
fare = data['Fare']


corr = scipy.stats.pearsonr(fare,Pclass)
#print(corr[0])
linear_model = scipy.stats.linregress(fare,Pclass)
slope = linear_model[0]
intercept = linear_model[1]
linear_fit = slope * fare + intercept

plt.figure()
plt.scatter(fare,Pclass)
plt.plot(fare,linear_fit, linewidth=3,label='Best fit')
plt.title('Passenger Class vs. Ticket Fare')
plt.xlabel('Fare')
plt.ylabel('Passenger Class')
plt.show()

pd.crosstab(data['Survived'], data['Pclass'])


#%%

fares_Q = data.query("Embarked=='Q'")['Fare']
fares_S = data.query("Embarked=='S'")['Fare']
fares_C = data.query("Embarked=='C'")['Fare']



Q_mean = fares_Q.mean()
S_mean = fares_S.mean()
C_mean = fares_C.mean()

print(Q_mean)
print(S_mean)
print(C_mean)






