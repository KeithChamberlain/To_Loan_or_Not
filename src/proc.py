#%%
'''
Imports
'''
import numpy as np
import pandas as pd
pd.options.display.max_rows=200

import os
import timeit as time
import matplotlib.pyplot as plt

# In order to correct plotting issue with plt.savefig()
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy import stats

'''
IO Functions
'''

def header_csv(file_path, delim):
    '''
    header_csv:
        wrapper for pd.read_csv. Reads from the file_path given,
        using the delimiter provided, from the header information
        only.
    returns
        Pandas Series object of headers read from the file.
    '''
    header = pd.read_csv(file_path, delimiter = delim, nrows=0)
    return header


def read_wc(file_path):
    try:
        return pd.read_csv(f"{file_path}.wc", 
        delimiter = " ", header=None).loc[0,1]
    except IOError:
        return "Path Not Found"

def resolve_wc(file_path):
    '''
    resolve_wc:
        Resolves the line count of a file. Wrapper for "wc -l"
        the OS shell is called for command wc (works on Linux)
    returns
        Tuple: number of lines in file if successful,
        "Path Not Found" otherwise; Time difference of system
        command.
    '''
    comnd = f"wc -l {file_path} > {file_path}.wc"
    print("System command: ", comnd)
    t0 = time.default_timer()
    os.system(comnd)
    t1 = time.default_timer() - t0
    return read_wc(file_path), round(t1,3)

def strip_delim(file_path, delims):
    '''
    strip_delims:
        Strips the delimiters from the file that get in the
        way of proper coersion to numeric datatypes, based 
        on the delimiters, such as % signs. Calls Linux OS
        commands.
    delims:
        A list of delimiters to strip from the file.
    Output:
        Outputs a new file with the delimiters stripped with
        the same file_path but with 2 appended at the end. 
    returns:
        file_path to new file.
    '''
    new_path_to_file = f"{file_path}.strip"
    comnd = f"cat {file_path} | tr -d '[\{delims}]' > {new_path_to_file}"
    print(comnd)
    t0 = time.default_timer()
    os.system(comnd)
    t1 = time.default_timer() - t0
    return new_path_to_file, round(t1, 3)

def ensure_skip0(rows:int):
    '''
    ensure_skip0:
        Ensures row number 0 is on the skip list so that the
        header isn't read in, causing all the columns to be
        read in as text.
    rows:
        an np.array() of rows to be added to. Usually from
        a random sample that may or may not have included the
        0th row.
    returns:
        np.array of rows including the 0th row
    '''
    new_list = [0]
    if not 0 in rows:
        new_list.extend(rows)
    return np.array(new_list)

def define_sample(rows:int, percnt:float=0.01, skip:bool=True, 
                    replace:bool=False, seed:int=None):
    '''
    define_sample:
        Instantiates and generates a random sequence based on a
        percentage of the data.
    percnt:
        The percentage of the data to generate a random sample of.
        Defaults to a 1% sample. 
    skip:
        percnt is taken as (1-percnt) if skip = True, such as in 
        the case of a 1% sample, 99% of the data need to be "skipped"
        at random.
    returns:
        sorted numpy array of rows.
    '''
    rndm = np.random.default_rng(seed)
    t0 = time.default_timer()
    if skip:
        result = rndm.choice(a=rows, size=rows - int(percnt*rows),
                             replace=replace)
    else:
        result = rndm.choice(a=rows, size=int(percnt*rows),
                            replace=replace)
    t1 = time.default_timer(), time.default_timer() - t0
    result = sorted(result)
    t2 = time.default_timer() - t1[0]
    print(f"Time to generate {percnt} random sample: ", round(t1[1], 3), 
          " seconds")
    print(f"Time to sort sample: ", round(t2, 3), " seconds")
    result = ensure_skip0(result)
    return result


def read_data(file_path, delim, skip, names):
    t0 = time.default_timer()
    df = pd.read_csv(file_path, delimiter=delim, header = None, 
                     skiprows = skip,
                     names = names, 
                     low_memory=False) #, nrows=10)
    t1 = round(time.default_timer() - t0,3)
    print(f"Time to read {file_path}: ", t1)
    return df

#%%
def plot_bar(num_overlays, dim_tup, x_list, heights_list, color_list, 
    xlab, ylab, title, suptitle, file_path, fontsize, labels):
    fig, ax = plt.subplots(1,1, figsize=dim_tup, sharex=True, 
        sharey=True, dpi=300)
    for _ in range(num_overlays):
        ax.bar(x=x_list[_], height=heights_list[_], 
            align="center", color=color_list[_], alpha=0.5,
            label=labels[_])
        ax.set_xlabel(xlab, fontsize=fontsize["x"])
        ax.set_ylabel(ylab, fontsize=fontsize["y"])
        ax.set_title(title, fontsize=fontsize["title"])
        ax.legend(fontsize=fontsize["legend"])
        plt.suptitle(suptitle, fontsize=fontsize["suptitle"])
    plt.show()
    fig.savefig(file_path)
    return fig, ax

#%%
def prob(splits_list, counts_list):
    probs=list()
    for _ in range(len(splits_list)):
        probs.append(np.sum(splits_list[_])/counts_list[_])
    return probs

def bin_means(counts_list, probs):
    binom_means=list()
    for _ in range(len(counts_list)):
        binom_means.append(counts_list[_]*probs[_])
    return binom_means

def bin_var(Ns, probs):
    binom_vars = list()
    for _ in range(len(probs)):
        binom_vars.append(Ns[_]*(probs[_]*(1-probs[_])))
    return binom_vars

def slow_stack(vector_a, vector_b, a_value, b_value):
    # print(vector_a.unique())
    # print(vector_b.unique())
    vector_a *= a_value
    vector_b *= b_value
    new_a = vector_a.copy().tolist()
    for _ in range(len(vector_b)):
        new_a.append(vector_b[_])
    return np.array(new_a)


#%%       

if __name__ == '__main__':
    '''
    Conditional setup
    '''
    emp_cutoff = 10
    loan_cutoff= 20000
    first_run = False
    full_data = False # then skip plots when not tested
    print(f"First run status for this script is {first_run}.")

    path_to_file1= '../data/accepted.csv' # Currently run from /src
    path_to_file2= '../data/rejected.csv' # Currently run from /src
    new_file2_path='../data/rejected.csv.strip' # run from /src
    print("The file_path to the data files 1 and 2 are as follows:")
    print(path_to_file1)
    print(path_to_file2)

    file1_length=0
    file2_length=0
    rand_sample = 1.0
    skip_sample = 1-rand_sample
    print(f"Using a {rand_sample*100}% random sample of data for EDA.")
    print(f"Skipping a {skip_sample*100}% sample of data for EDA.")
    
    '''
    PLOT PARAMETERS
    '''
    pad_inches = 2


    '''
    FIRST RUN CONDITIONAL
    '''
    if first_run:
        file1_length_tup = resolve_wc(path_to_file1)
        file2_length_tup = resolve_wc(path_to_file2)
        print("File Length file 1 and time taken: ", file1_length_tup,
                " seconds")
        print("File length file 2 and time taken: ", file2_length_tup,
                " seconds")

        '''
        File 2 has % sign in a numeric field
        preventing it from becoming numeric.
        This call strips the extra delimiter.
        '''
        # Note this isn't generalized for multiple delimiters yet.
        new_file2_path_tup = strip_delim(path_to_file2, "%")
        print("The new path and timing is: ", new_file2_path_tup,
                " seconds.")
    else:
        file1_length = read_wc(path_to_file1)
        file2_length = read_wc(path_to_file2)
        print("File Lengths: ", file1_length, file2_length)
#%%
    '''
    READ DATA
    '''
    file1_header = header_csv(path_to_file1, ",")
    file2_header = header_csv(path_to_file2, ",")
    accskip = define_sample(file1_length, rand_sample) # accept skip rows
    rejskip = define_sample(file2_length, rand_sample) # reject skip rows
    accepted = read_data(path_to_file1, ",", accskip,
                          file1_header.columns) # accepted data
    rejected = read_data(new_file2_path, ",", rejskip, 
                         file2_header.columns) # rejected data

#%% 
    ''' 
    Recode Employment Length
    '''
    names = {"< 1 year": 0, "1 year": 1,  "2 years": 2, "3 years": 3,
             "4 years": 4,  "5 years": 5,"6 years": 6,   
            "7 years": 7, "8 years": 8, "9 years": 9, 
            "10+ years": 10,  np.nan: np.nan}
    rejected["Employment Length"].replace(to_replace=names, 
        inplace=True)
    accepted["emp_length"].replace(to_replace=names, 
        inplace=True)
#%%    
    rejected["Employment Length"].value_counts(sort=
        False).plot(kind="bar")
    plt.show()
    accepted["emp_length"].value_counts(sort=
        False).plot(kind="bar")
    plt.show()
    '''
    Plot Employment Length
    '''
    if not full_data:
        rejtotl = rejected["Employment Length"].count()
        acctotl = accepted["emp_length"].count() 
        h1 = accepted["emp_length"].value_counts(sort=
            False)/acctotl
        h2 = rejected["Employment Length"].value_counts(sort=
            False)/rejtotl
        x2 = [0,5,7,4,1,10,9,3,2,8,6]
        x1 = [0,2,3,6,5,9,7,4,8,1,10]
                
        fig1, ax1 = plot_bar(num_overlays=2, dim_tup=(10,5), 
            x_list=[x1,x2], 
            heights_list=[h1,h2], color_list=["blue", "orange"], 
            xlab = "Length of Employment (years)", 
            ylab="Percent (%)", 
            title="Distribution of Length of Employment", 
            suptitle=None, file_path="../img/barplt.png",
            fontsize={"x":25,"y":25,"xticks":20,"yticks":20,
                "title":30, "suptitle":None,"legend":15},
            labels=["Accepted Loans","Rejected Loans"])
        fig1.savefig("../img/emp_len_bar.png")

#%%
    '''
    Plot Loan Amounts Requested
    '''
    if not full_data:
        fig2, ax2 = plt.subplots(1,1, figsize=(10,5), sharex=True, sharey=True)
        ax2.hist(accepted["loan_amnt"][accepted["loan_amnt"]<42000], 
            bins=50, alpha=0.5, density=True, color="blue", label="Accepted Loans")
        ax2.hist(rejected["Amount Requested"][rejected["Amount Requested"]<42000], 
            bins=50, alpha=0.5, density=True, color = "orange", label="Rejected Loans")
        ax2.set_xlabel("Loan Amount Requested", fontsize=25)
        ax2.set_ylabel("Density", fontsize=25)
        ax2.set_title("Distribution of Loan Amounts Requested", fontsize=30)
        ax2.legend(fontsize=15)
        plt.show()
        fig2.savefig("../img/loan_amt_hist.png")

# %%
    '''
    Grab binomial splits of interesitng cutoffs.
    Add to df dataframe in 0's, and 1's
    '''
    Emp_lty = 0*(rejected["Employment Length"] < emp_cutoff) + \
        1*(accepted["emp_length"] < emp_cutoff)
    Emp_gtey= 0*(rejected["Employment Length"]>=emp_cutoff) + \
        1*(accepted["emp_length"]>=emp_cutoff)
    LoanReq_ltk = 1*(accepted['loan_amnt']<loan_cutoff) + \
        0*(rejected["Amount Requested"]<loan_cutoff)
    LoanReq_gtek= 1*(accepted['loan_amnt']>=loan_cutoff) + \
        0*(rejected["Amount Requested"]>=loan_cutoff)

    splits_list = [Emp_lty,Emp_gtey, LoanReq_ltk,LoanReq_gtek]

#%%
    '''
    Calculate probabililities, means, and variance
    for each group.
    '''
    El_totl = Emp_lty.count()
    Eg_totl = Emp_gtey.count()
    Ll_totl = LoanReq_ltk.count()
    Lg_totl = LoanReq_gtek.count()
    counts_list = [El_totl,Eg_totl,Ll_totl,Lg_totl]

    # P = ones/total
    probs = prob(splits_list, counts_list) # left to right
    # mu = n*P
    binom_means = bin_means(counts_list, probs)
    # var = n*p*(1-p)
    binom_vars = bin_var(counts_list, probs)
    print(counts_list, binom_vars, binom_means, probs)

#%%
    '''
    Plot Two Group Plots, just the successes
    '''
    new_vec = slow_stack(np.array(Emp_lty[Emp_lty==1]), 
        np.array(Emp_gtey[Emp_gtey==1]), -.5, .5)
    new_vec2 = slow_stack(np.array(LoanReq_ltk[LoanReq_ltk==1]), 
        np.array(LoanReq_gtek[LoanReq_gtek==1]), -.5, .5)
    new_vec = pd.Series(new_vec)
    new_vec2 = pd.Series(new_vec2)
    plt.show()
    plt.show()

    fig0=new_vec.value_counts().plot(kind="bar", 
        label="Years Employed", color="blue", alpha=0.5, x=[""])
    plt.xlabel("Years Employed", fontsize=25)
    plt.ylabel("Frequency", fontsize=25)
    plt.title("Years Employed", fontsize=30)
    plt.xticks(ticks=[0,1], labels=["[0, 10)","10+"], rotation = 0)
    plt.savefig("../img/Emp_years_approv.png",dpi=300, pad_inches=pad_inches)
    plt.show()


    figb=new_vec2.value_counts().plot(kind="bar", label="Amount",
        color="blue", alpha=0.5)
    plt.xlabel("Loan Amount",fontsize=25)
    plt.ylabel("Frequency",fontsize=25)
    plt.title("Loan Amount", fontsize=30)
    plt.xticks(ticks=[0,1], labels=["(\$0,\$20k)", "\$20k+"], rotation = 0)
    plt.savefig("../img/Amt_approv.png",dpi=300, pad_inches=pad_inches)
    plt.show()

# %%
    '''
    Stating that the null hypothesis for a two sample test of approximate 
    population proportions for Accepted and Rejected Loans between applicants with 
    10 or more years of employement history and appicants with 9 or less years of 
    employment history is that there is no difference betweeen their proportion of 
    accepted loans between the groups. My hypothesis is that the group with ten or
    more years of employment will have a statistically different number of 
    accepted loans than those with less than ten years of employment, on 
    average. 
    '''
    counts_list=pd.Series(counts_list)
    probs = pd.Series(probs)

    binom_means=np.array(binom_means)
    binom_vars=np.array(binom_vars)
    counts_list=np.array(counts_list)


    # Each binomial random variable
    bnml_emp_lt10 = stats.binom(n=counts_list[0], p=probs[0])
    bnml_emp_gte10= stats.binom(n=counts_list[1], p=probs[1])
    bnml_loan_lt20= stats.binom(n=counts_list[2], p=probs[2])
    bnml_loan_gte20= stats.binom(n=counts_list[3], p=probs[3])

    '''
    Near as I can tell, I can't do a test of two population proportions 
    with unequal n in each group with binomial ditributions, which would just be
    Pa - Pb ~ 0 with summed n. I need to utilize the normal approximation. 
    To calculate the sample standard error for population proportions, I use 
    the formula:
        SE(P) = sqrt(P(1-P)(1/n_a + 1/n_b)) | P = (Pa+Pb) or the total proportion
    '''
    def se_two_prop(Ptotal, na, nb):
        return np.sqrt(Ptotal*(1-Ptotal)*(1/na+1/nb))

    def z_score_prop(best_estimate, Ho, SE):
        return (best_estimate-Ho)/SE
    
    se_emp = se_two_prop(Ptotal = probs[0:1].sum(), na = counts_list[0],
        nb = counts_list[1])
    se_loan = se_two_prop(Ptotal = probs[2:3].sum(), na = counts_list[2], 
        nb = counts_list[3])
    z_emp = z_score_prop(best_estimate  = probs[0] - probs[1], Ho = 0, 
        SE = se_emp)
    z_loan = z_score_prop(best_estimate = probs[2] - probs[3], Ho = 0, 
        SE = se_loan)
    p_emp = stats.norm.cdf(x=z_emp, 
        loc = np.abs(probs[0]*counts_list[0]-probs[1]*counts_list[1]), 
        scale = se_emp)
    p_loan = stats.norm.cdf(x=z_loan,
        loc = np.abs(probs[2]*counts_list[2] - probs[3]*counts_list[3]), 
        scale = se_loan)
    stringz = f'''
    "The z-test statistic for Employment Years is : {z_emp}, with a p-value of 
    {p_emp}.
    
    The z-test statistic for Loan Amount is : {z_loan}, with a p-value of 
    {p_loan}.
    '''

    print(stringz)


    t_score0, p0 = stats.ttest_ind(Emp_lty, Emp_gtey, equal_var=False, 
    nan_policy="omit")

    t_score, p = stats.ttest_ind(LoanReq_ltk, LoanReq_gtek, equal_var=False, 
        nan_policy="omit")

    string1=f'''
    The t-test statistic for Employment Years is {t_score0}, with a p-value of 
    {p0}. 

    The t-test statistic for Loan Amount is {t_score}, with a p-value of {p}.
    '''
    print(string1)


#%%

LoanReq_ltk
# %%
print(np.isnan(accepted["loan_amnt"]).sum())
print(np.isnan(rejected["Amount Requested"]).sum())
print(np.isnan(accepted["fico_range_low"]).sum())
print(np.isnan(accepted["fico_range_high"]).sum())
print(np.isnan(rejected["Risk_Score"]).sum())
print(np.isnan(accepted["dti"]).sum())
print(np.isnan(rejected["Debt-To-Income Ratio"]).sum())
print(np.isnan(accepted["emp_length"]).sum())
print(np.isnan(rejected["Employment Length"]).sum())
# print(np.isnan(accepted["zip_code"]).sum())
# print(np.isnan(rejected["Zip Code"]).sum())
# print(np.isnan(accepted["addr_state"]).count())
# print(np.isnan(rejected["State"]).count())

#%%
    print(binom_means[1]-binom_means[0])
    print(binom_means[3]-binom_means[2])
    

# %%
