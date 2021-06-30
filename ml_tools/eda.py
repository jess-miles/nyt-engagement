import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data_cont(to_explore, df, target, hist=True, box=True, plot_v_target=True,
                 summarize=True, norm_check=True):
    """Creates plots and summary information intended to be useful in preparing
    for linear regression modeling. 
    Prints plots of distributions, a scatterplot of each predictor column against 
    a target, and outputs a dataframe of metadata including results of a normality 
    check and correlation coefficient.

    This function works best with a continuous target variable, although predictors
    may be categorical.

    Returns: a dataframe representing the metadata collected.
    _____________________________
    Args:
    -----------------------------
    
    to_explore: list of column names to explore
    
    df: Dataframe containing the columns in to_explore, as well as the target
        column
    
    target: string of the column name to use as the target, or dependent variable
    
    hist: True or False (default True). Whether to include a histogram for each
    predictor column.
    
    box: True or False (default True). Whether to include a box plot for each
    predictor column.
    
    plot_v_target: True or False (default True). Whether to include a scatter 
    plot showing the predictor versus target
    
    summarize: True or False (default True). Whether to include a summary of
    the values in each predictor column. Data will be summarized using 
    df.describe() for variables deemed continuous, and df.sort_values()
    for variables deemed categorical. Classification of continuous versus
    categorical is best effort.
    
    norm_check: True or False (default True). Whether to perform a normality
    check using SciPy's stats omnibus normality test. Null hypothesis 
    is that the data comes from a normal distribution, so a value less than
    0.05 represents likely NOT normal data.
    """
    
    # Create some variables to dynamically handle including/excluding 
    # certain charts
    num_charts = 0
    if hist:
        num_charts += 1
    if box:
        num_charts += 1 
    if plot_v_target:
        num_charts += 1
        
    # check if input column is a list; if not, make it one. This allows for
    # a string to be passed if only one column is being summarized.
    if type(to_explore) == str:
        temp_list = [to_explore]
        to_explore = temp_list
    
    # column headers for metadata output df
    meta_list = [['col_name', 'corr_target', 'assumed_var_type', 'omnibus_k2',
                 'omnibus_pstat', 'is_normal', 'uniques', 'mean', 'median']]
    
    # loop through each column in the list to analyze
    for col in to_explore:
        
        header_line = '-'*75
        header_text = f'\nExploring column: {col}\n'
        print(header_line + header_text + header_line)
        
        # Determine if categorical or continuous
        # assume continuous to begin with
        var_type = 'continuous'
        data_type = df[col].dtype
        uniques = np.nan
        mean = np.nan
        median = np.nan
        num_uniques = len(df[col].unique())
        
        if df[col].dtype in ['int64', 'float64']:
            # number types need the most analysis because they could be
            # categorical even if they're numeric
            
            # using 100 as an arbitrary cutoff here, may need adjustment
            if num_uniques < 20:
                var_type = 'categorical'
                uniques = num_uniques
            else:
                mean = np.mean(df[col])
                median = np.median(df[col])
        elif df[col].dtype in ['object']:
            # Assuming column types have been fixed at this point,
            # so if a column is not numerical it must be categorical
            var_type = 'categorical'
            uniques = num_uniques
        elif df[col].dtype in ['datetime64']:
            var_type = 'date'
            
        # print summary based on data type
        if summarize:
            if var_type in ['continuous', 'date']:
                header_text = f'\ndf.describe() for continuous data: {col}\n'
                print(header_line + header_text + header_line)
                print(df[col].describe())
            else:
                header_text = f'\nValue Counts for categorical data: {col}\n'
                print(header_line + header_text + header_line)
                with pd.option_context('display.max_rows', 20):
                    print(df[col].value_counts())
        
        # creates scatter plots, histogram, and box plots for numerical data
        if data_type in ['int64', 'float64']:
            if num_charts > 0:

                fig, axes = plt.subplots(nrows=num_charts, ncols=1, 
                                         figsize=(8, num_charts * 5))
                if hist:
                    if num_charts > 1:
                        ax1 = axes[0]
                    else:
                        ax1 = axes

                    if box:
                        ax2 = axes[1]
                        if plot_v_target:
                            ax3 = axes[2]
                    elif plot_v_target:
                        ax3 = axes[1]
                elif box:
                    if num_charts > 1:
                        ax2 = axes[0]
                    else:
                        ax2 = axes

                    if plot_v_target:
                        ax3 = axes[1]

                elif plot_v_target:
                    ax3 = axes


                # add a little extra space for headers
                plt.subplots_adjust(hspace=0.3)

                # Histogram
                if hist:
                    sns.histplot(df[col], kde=True, ax=ax1)
                    ax1.set_title(f"Hist {col}")

                # Box plot
                if box:
                    sns.boxplot(x=df[col], ax=ax2)
                    ax2.set_title(f"Boxplot {col}")

                # Plot against target
                # create a series representing quartiles, to use as hue
                if plot_v_target:                
                    if var_type == 'continuous':
                        try:
                            quartile_labels=['q1', 'q2', 'q3', 'q4']
                            quartiles = pd.qcut(df[col], 4, 
                                                labels=quartile_labels, 
                                                duplicates='drop')
                            sns.scatterplot(x=df[col], y=df[target], ax=ax3, 
                                            hue=quartiles)
                            ax3.legend(title=f'{col} quartiles')
                            
                        except:
                            sns.scatterplot(x=df[col], y=df[target], ax=ax3)
                    else:
                        sns.scatterplot(x=df[col], y=df[target], ax=ax3)
                    ax3.set_title(f"{col} versus {target}")
                    
                plt.show();
                
            # get pearson correlation coefficient between col and target
            corr = df[[col, target]].corr()

            # Test for normality using scipy omnibus normality test
            # null hypothesis is that the data comes from a normal distribution
            if norm_check:
                k2, p = stats.normaltest(df[col])
                if p < 0.05:
                    normal = False
                    print(f'\nData is NOT normal with p-statistic = {p}\n')
                else:
                    normal = True
                    print(f'\nData IS normal with p-statistic = {p}\n')

            # append metadata to list of lists
            meta_list.append([col, corr.iloc[0][1], var_type, k2, p, normal, 
                uniques, mean, median])
            
        # Create catplot for categorical data
        elif data_type in ['object', 'str', 'category']:
            # get variable to determine appropriate height based on number
            # of categories to be displayed
            h = len(df[col].value_counts())
            
            # Get list of categories sorted in alpha order
            order = df[col].unique()
            order.sort()
            
            fig, ax = plt.subplots(figsize=(8, (h*0.15)+4))
            sns.barplot(x=target, y=col, data=df, orient='h', 
                        order=order, ax=ax)
            ax.set_title(f"Average {target} per {col}");
            plt.show();
        
    df_meta = pd.DataFrame(data=meta_list[1:], columns=meta_list[0])
    return df_meta

def explore_data_catbin(to_explore, df, target, pred_type='cat'):
    """
    Generates visualizations to explore the relationship between predictors
    and a binary categorical target. Specify the type of predictors using the
    `pred_type` parameter: accepted values are `cat` for categorical and 
    `cont` for continuous.
    
    This function assumes a binary target.
    
    """
    if pred_type not in ['cat', 'cont']:
        print("Error: `pred_type` should be 'cat' for categorical\
        predictors and 'cont' for continuous predictors. No other\
        values accepted.")
        return None
    
    disc = True if pred_type == 'cat' else False
    
    # get mean of target. Since target is binary, mean is representative
    # of the proportion of 1 labels to 0 labels
    pop_mean = np.round(df[target].mean(), 4)
    
    # draw plots
    for col in to_explore:
        fig, [ax1, ax2] = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
        plt.tight_layout(pad=3)

        sns.histplot(data=df, x=col, ax=ax1, discrete=disc)
        ax1.set_title(f"Distribution of {col}")

        if pred_type=='cat':

            sns.pointplot(data=df, x=col, y=target, ci=68, ax=ax2, join=False,
                         scale=1.5, capsize=0.05)
            ax2.set_title("Target Mean per Category")
            ax2.axhline(pop_mean, color='red', ls='dashed', label='population mean')
            ax2.legend();
            
        elif pred_type=='cont':
            
            sns.boxenplot(data=df, x=col, y=target, ax=ax2, orient='h', width=1)
            ax2.set_title("Feature Distribution Per Target Class");
            
    return None



def currency(x, pos=None):
    """Formats numbers as currency, including adding a dollar sign and abbreviating numbers
    over 1,000. Can be used to format matplotlib tick labels.
    _____________________________
    Args:
    -----------------------------
        x (integer or float): Number to be formatted.
        pos (optional): Included for matplotlib, which will use it. Defaults to None.

    Returns:
        string: formatted string based on number.
    """
    # over 1 billion
    if abs(x) >= 1000000000:
        return '${:1.2f} B'.format(x*1e-9)
    # over 10 million
    elif abs(x) >= 10000000:
        return '${:1.1f} M'.format(x*1e-6)
    # over 1 million
    elif abs(x) >= 1000000:
        return '${:1.2f} M'.format(x*1e-6)
    elif x == 0:
        return '${:0}'.format(x)
    elif abs(x) >= 1000:
        return '${:1.1f} K'.format(x*1e-3)
    else:
        return '${:.1f}'.format(x)
