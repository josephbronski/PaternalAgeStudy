import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from factor_analyzer import FactorAnalyzer
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats as stats
import statsmodels.api as sm

def rChart(x, y, xname, yname, noise=False, u=0, sd=1, size=5, lowessReg=True):
    # Calculate the correlation coefficient and the p-value
    r, p = stats.pearsonr(x, y)

    # Calculate the sample size
    n = len(x)


    # Compute the 95% confidence interval for r
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-(1-0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    if (lowessReg == True):
        # Add the LOESS regression line
        lowess = sm.nonparametric.lowess
        z = lowess(y, x, frac=1./3, it=0)  # you can modify frac and it as needed
        plt.plot(z[:, 0], z[:, 1], color='blue')

    # Add the line of best fit
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red')
    
    if(noise==True):
        noise1 = np.random.normal(loc=u, scale=sd, size=len(x))
        noise2 = np.random.normal(loc=u, scale=sd, size=len(x))
        plt.scatter(x + noise1, y + noise2, s=size, color="black")     
    else:
        # Create the scatter plot
        plt.scatter(x, y, s=size, color="black")


    # Add the correlation coefficient, 95% CI, and sample size to the plot
    plt.text(0.05, 0.95, f'r = {r:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}], n = {n}', transform=plt.gca().transAxes)
    # Add labels and title
    plt.xlabel(xname)
    plt.ylabel(yname)

    # Display the plot
    plt.show()


def histograms(x, y, xname, yname, varname):
    mean_0 = x.mean()
    std_0 = y.std()

    mean_1 = y.mean()
    std_1 = y.std()
    d = (mean_0 - mean_1) / ( (std_0 + std_1)/2)
    t_stat, p_value = stats.ttest_ind(x, y)
    # Create a histogram for the group where binary_col is 0
    x.hist(alpha=0.5, bins=30, label='1. ' + xname)

    # Create a histogram for the group where binary_col is 1
    y.hist(alpha=0.5, bins=30, label='2. ' + yname)

    # Add labels
    plt.xlabel(varname)
    plt.ylabel('Frequency')
    plt.text(0.68, 0.80, f'd = {d:.3f}, p = {p_value:.3f}', transform=plt.gca().transAxes)
    plt.text(0.68, 0.75, r'$\mu_1$ = {:.2f}, $\mu_2$ = {:.2f}'.format(mean_0, mean_1), transform=plt.gca().transAxes)
    plt.text(0.68, 0.70, r'$n_1$ = {}, $n_2$ = {}'.format(len(x), len(y)), transform=plt.gca().transAxes)
    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

def groupRchart(df, groupby, group, names=None, title='Plot', n=1, exclude=None):
    #df is the dataframe with the data
    #groupby are the things to clump eg ages
    #group is the thing to get the mean of by each groupby. Eg the mean IQ by age
    #names are a tuple if you want to change axis names. (xname, yname)
    #title changes plot title
    #n groups by every n of groupby. So 2 would be it groups by every 2 years. 10 every decade etc. 
    #exclude is a list of values to exclude eg [10, 20, 30, 50]
    xname = groupby
    yname = group
    if (names != None):
        xname, yname = names
    # 1. Create a new column that bins the age into decades
    df['decade'] = (df[groupby] // n) * n
    if exclude is not None:
        df = df[~df['decade'].isin(exclude)]

    # 2. Group by the new decade column and compute the mean of the binary column
    df_decade = df.groupby('decade')[group].agg(['mean', 'sem']).reset_index()

    # Calculate the margins (1.96 * standard error)
    df_decade['lower'] = df_decade['mean'] - 1.96 * df_decade['sem']
    df_decade['upper'] = df_decade['mean'] + 1.96 * df_decade['sem']

    # Calculate line of best fit
    slope, intercept = np.polyfit(df_decade['decade'], df_decade['mean'], 1)

    # Calculate correlation coefficient (r value)
    r_value = np.corrcoef(df_decade['decade'], df_decade['mean'])[0, 1]
    
    # Compute the 95% confidence interval for r
    r_z = np.arctanh(r_value)
    se = 1/np.sqrt(df_decade['decade'].size-3)
    z = stats.norm.ppf(1-(1-0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))

    # Plot the means by decade in a scatter plot with error bars
    plt.errorbar(df_decade['decade'], df_decade['mean'], yerr=(df_decade['mean']-df_decade['lower'], df_decade['upper']-df_decade['mean']), fmt='o')
    plt.plot(df_decade['decade'], slope * df_decade['decade'] + intercept, color='red')  # add line of best fit
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)

    # Add r value and equation to the plot
    plt.text(0.05, 0.95, f'r = {r_value:.2f}, 95% CI = [{lo:.2f}, {hi:.2f}]', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'y = {slope:.2f}x + {intercept:.2f}', transform=plt.gca().transAxes)

    plt.grid(True)
    plt.show()

def groupRchart2(data, groupby, group, explicit=True, names=None, title='Plot', n=1, exclude=None, labels=None):
    #df is the dataframe with the data
    #groupby are the things to clump eg ages
    #group is the thing to get the mean of by each groupby. Eg the mean IQ by age
    #names are a tuple if you want to change axis names. (xname, yname)
    #title changes plot title
    #n groups by every n of groupby. So 2 would be it groups by every 2 years. 10 every decade etc. 
    #exclude is a list of values to exclude eg [10, 20, 30, 50]
    xname = groupby
    yname = group
    colors = ['blue', 'orange']
    if (labels is None):
        labels = ['<35 fathers', '>35 fathers']
    if (names != None):
        xname, yname = names
    if data is not None:
        for df, c, l in zip(data, colors, labels):
            # 1. Create a new column that bins the age into decades
            df['decade'] = (df[groupby] // n) * n
            if exclude is not None:
                df = df[~df['decade'].isin(exclude)]

            # 2. Group by the new decade column and compute the mean of the binary column
            df_decade = df.groupby('decade')[group].agg(['mean', 'sem']).reset_index()

            # Calculate the margins (1.96 * standard error)
            df_decade['lower'] = df_decade['mean'] - 1.96 * df_decade['sem']
            df_decade['upper'] = df_decade['mean'] + 1.96 * df_decade['sem']

            # Calculate line of best fit
            slope, intercept = np.polyfit(df_decade['decade'], df_decade['mean'], 1)

            # Calculate correlation coefficient (r value)
            r_value = np.corrcoef(df_decade['decade'], df_decade['mean'])[0, 1]

            # Compute the 95% confidence interval for r
            r_z = np.arctanh(r_value)
            se = 1/np.sqrt(df_decade['decade'].size-3)
            z = stats.norm.ppf(1-(1-0.95)/2)
            lo_z, hi_z = r_z-z*se, r_z+z*se
            lo, hi = np.tanh((lo_z, hi_z))

            # Plot the means by decade in a scatter plot with error bars
            if (explicit is True):
                plt.errorbar(df_decade['decade'], df_decade['mean'], yerr=(df_decade['mean']-df_decade['lower'], df_decade['upper']-df_decade['mean']), fmt='o', label=l, color=c)
            plt.plot(df_decade['decade'], slope * df_decade['decade'] + intercept,)  # add line of best fit
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.title(title)
            
    plt.legend()  # add this line

    plt.grid(True)
    plt.show()

