import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features

# Making a data frame
filename = "credit_card_data_set.xls"
nanDict = {}
CDdf = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
CDdf.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

class Dataset(pd.DataFrame):
    '''Divide the DataFrame into data and target.'''
        
    def polish_and_divide(self, targetcol = -1, headerrows = 0, headercols = 0):
        values = np.copy(self.to_numpy())
        if headerrows == 0:
            self.feature_names = np.copy(self.columns)
        else:
            self.feature_names = self.iloc[headerrows-1]
        self.data = values[headercols: , headerrows : targetcol]
        self.target = values[headercols: , targetcol]
        

#Plot bar chart for the different features. Notice the results without labels (0 in gender, education and so on)
#Give name to these categories in each feature.
print(CDdf['AGE'].max())
print(CDdf['AGE'].min())
    
CDds = Dataset(CDdf)
CDds.polish_and_divide(headerrows = 0, headercols = 1)

target = CDds.target
data = CDds.data
featnames = CDds.feature_names

plot_features(CDds)

fig = plt.subplots()
import seaborn as sns
correlation_matrix = CDds.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()