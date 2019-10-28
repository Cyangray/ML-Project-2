import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Making a data frame
filename = "credit_card_data_set.xls"
nanDict = {}
CDdf = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
CDdf.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

class Dataset(pd.DataFrame):
    '''Divide the DataFrame into data and target.'''
        
    def polish_and_divide(self, targetcol = -1, headerrows = 0, headercols = 0):
        values = self.to_numpy()
        if headerrows == 0:
            self.feature_names = self.columns
        else:
            self.feature_names = self.iloc[headerrows-1]
        self.data = values[headercols: , headerrows : targetcol]
        self.target = values[headercols: , targetcol]
    
CDds = Dataset(CDdf)
CDds.polish_and_divide(headerrows = 0, headercols = 1)

target = CDds.target
data = CDds.data
featnames = CDds.feature_names

fig, axes = plt.subplots(11,2,figsize=(10,20))
non_default = CDds.data[CDds.target == 0]
default = CDds.data[CDds.target == 1]
ax = axes.ravel()

for i in range(22):
    _, bins = np.histogram(CDds.data[:,i], bins =50)
    ax[i].hist(non_default[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(default[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(CDds.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Non-default", "default"], loc ="best")
fig.tight_layout()
plt.show()

import seaborn as sns
correlation_matrix = CDds.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

#print eigvalues of correlation matrix
EigValues, EigVectors = np.linalg.eig(correlation_matrix)
#print(EigValues)