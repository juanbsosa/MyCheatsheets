#PYTHON

# %% VISUAL STUDIO CODE

# KEYBOARD SHORTCUTS

# Access the Command Palette
CTRL+SHIFT+P

# Save the workspace image so you can reopen the same files as in your last session:
file -> Add folder to workspace -> Save Workspace As

# Select multiple lines at the same time:
alt+shift

# Wrap text (make text adjust to the window size):
alt+z

# Select a word and then select all the repetitions of that word:
ctrl + d

# Enable the option to find and replace in a selection in Visual Studio Code:
download Quick Replace In Selection extension

# Show/visualize the entire database with all columns (to view in Visual Studio Code or other tools):
pd.set_option('display.max_columns', None)
pd.set_option('max_row', None) #I don't know what this one does

# Fold all code chunks
ctrl+k + ctrl+0

# %% DATACAMP COURSES starting from here...

# %% BASIC COMMANDS

# Look for the position of an element in a list/dictionary/etc according to its value
x=["a","b"]
x.index("b")

# Create a list of numbers from 1 to n
list(range(1,5))

# Create a dictionary from two lists
list_a=["a", "b", "c"] # keys
list_b=[[1,2,3], ['asd', 'ds'], [123]] # values per key
dict(zip(list_a, list_b))
# Another way
v1 = [0.01, 0.001, 0.0001] # values 1
v2 = [100, 150, 200] # values 2
dict(key1=v1, key2=v2)

# %% DICTIONARIES
a= {"saludo":"hola"}

# Get keys
a.keys()

# Get values
a.values()

# Get specific value using key
a.get('saludo')

# Add/change a key-value pair
a["saludar"] = "holar"

# Delete key-value pairs
del(a["saludar"])

# Check if value is part of keys
"saludar" in a

# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }
    # Use chained square brackets
print(europe["france"]["capital"])

# %% LOOPS

# for lop

# pair index with value
fam =[1,3,5,1]
for index, value in enumerate(fam):
    print("index " + str(index) + ": " + str(value))
    
# Iterate through key-value pairs in a dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
for country, capital in europe.items():
    print(country, " ", capital)
    
# Iterate over elements of a 2D numpy array
for x in np.nditer(my_array) :
    ...

# Continue code in the next line
df.groupby("col") \
    .mean()

# %% NUMPY - ARRAYS

# Logical equivalents of AND and OR and NOT
np.logical_and()
np.logical_or()
np.logical_not()

# %% NUMPY - RANDOM NUMBERS
from numpy import random

# Set seed
random.seed(123)
np.random.seed(123)

# %% PANDAS LIBRARY - DATA FRAMES
import pandas as pd

# Build a DataFrame from a dictionary of lists (column by column)
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)

# Build a DataFrame from a list of dictionaries (row by row)
list_of_dicts[{"country":"United States", 'drives_right':"True", 'cars_per_cap':809},
{"country":"Australia", 'drives_right':"False", 'cars_per_cap':731}]
cars=pd.DataFrame(list_of_dicts)

# Access/change row index
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels

# Number of rows of a data frame
len(df)

# Read a csv file
pd.read_csv("file.path") # index_col to say which column corresponds to row index

# Write a csv file / convert data frame to csv
df.to_csv("file.csv")

# Select a column of a data frame (several ways)
cars["cars_per_cap"] # returns a pandas series
type(cars["cars_per_cap"])

cars[["cars_per_cap"]] # returns a pandas (sub) data frame
type(cars[["cars_per_cap"]])

# Select multiple columns of data frame
df[["col1, col2"]]
#or
cols = ["col1, col2"]
df[cols]

# Select rows by index
cars[0:5]

# Slice by rows/columns

# Select rows by row names
cars.loc["US"]
cars.loc[["US", "RU"]]

# Select rows by row names and column names
cars.loc[["RU"], ["country"]]

# Select all rows from columns
cars.loc[:, ["country", "drives_right"]]

# Subset data frame based on positions of rows/columns
cars.iloc[[1, 4]]
cars.iloc[[1, 4], [0, 1]]
cars.iloc[:, [0, 1]]

# Iterate over column names of data frame
for col in cars:
    print(col)

# Iterate over rows of data frame (row labes and row data)
for lab, row in cars.iterrows():
    print(lab)
    print(row)
    
# AAPLY METHOD: pass a function to a column of  data frame (it can be a lambda function)
# Create new columns by using apply on a function
    # eg the length of the name of the country
cars["length_country_name"] = cars["country"].apply(len)
cars

# Display column names, data types and missing values
df.info()

# Data frame dimension attribute
df.shape

# Overview of descriptive statistics of data frame
df.describe()

# Values of data frame (attribute) / get data frame values as array / convert the column of a data frame to a numpy array
df.values

# Columns of data frame (attribute)
df.columns

# Row names / row indexes (attribute)
df.index

# Get data types of all  columns of a data frame
df.dtypes

# Sort rows of data frame
df.sort_values("column_name")
df.sort_values("column_name", ascending=False)
# Sort data frame accoridn to multiple columns
df.sort_values(["column_name", "col_name2"], ascending=[True, False])

# Subsetting according to logicla condition
df[df["col"]>10]
df[df["col"]=="value"]
df[(df["col"]>10) & (df["col"]=="value")] # !!! add parenthesis

# Subsetting using .isin()
df[df["col"].isin(["v1", "v2"])]

# Create a new column
df["col"] = df["col2"]*2

## SUMMARY STATISTICS
df["col"].mean() # mean of column
df["col"].median() # median of column
df["col"].mode() # mode of column
df["col"].min() # minimum of column
df["col"].max() # maximum of column
df["col"].var() # variance of column
df["col"].std() # standard deviation of column
df["col"].sum() # sum of column
df["col"].sum(axis='columns') # sum by row
df["col"].quantile() # quantile of column
df["col"].cumsum() # cumulative sum of column
# NOTE: this are all calculated based on the "index" axis of the data frame, which means "by rows", because of the default value
df.mean(axis="index")
# But you can also calculate summary statistics for each row (across columns)
df.mean(axis="columns")
# If you dont specify any column, the operation is computed over all columns
df.mean()

# Custom summary statistics using ".agg()" method (example: get 30th percentile)
def pct30(column):
    return column.quantile(0.3)
df["col"].agg(pct30) # there can be more than one function as arguments
df["col"].agg([pct30, median])

## COUNTING
# Count unique values of a column
df["col"].value_counts()
# count unique values and sort by frequency in descending order
df["col"].value_counts(sort=True)
# count unique values and sort by value of the tabulated variable
df["col"].value_counts.sort_index()
# Get the most frequent value of a column (similar to .mode())
df['col'].value_counts().index[0]
# turn the counts into proportions of the total
df["col"].value_counts(normalize=True)
# count unique values and missing values as well
df["col"].value_counts(dropna=False) # otherwise, the function excludes missing values by default
# Count by groups
df.groupby(['col1']).agg({'col2': 'count'}).reset_index()
# Create a bar plot of the frequencies for unique values
df["col"].value_counts(sort=False).plot(kind='bar') # sort False is to preserve the original order and not sort by frequency

# Keep rows according to the number of obsevations by group
nobel.groupby('col').filter(lambda group: len(group) >= 2)

# Drop duplicates
df.drop_duplicates(subset=["column1", "column2"])

# Group by column
df.groupby(["col1", "col3"])["col2", "col4"].agg([np.min, np.max, np.mean, np.median])
df.groupby("col").agg({'col2':'count'})
    # Group by index (when you have a multiindex)
df.groupby(level=0).agg({'col2':'count'}) # first index
# Group by and get the first element by group
df.groupby('var').first()
# Group by without setting the group variable as index
df.groupby("col", as.index=False)

# When you group by two columns, the result is a MultiIndex Pandas Series
# Convert a multi index pandas series to a data frame
data_with_mindex.unstack()
# Although you can process it like a data frame...
df.loc['value_index1', 'value_index2']
# Instead of using a groupby and an unstack, one could directly use the 'pivot_table' method
df.pivot_table(index='col1', columns='col2', values='col3')

# Pivot tables
df.pivot_table(values="col1", index="col_group") # mean by default
df.pivot_table(values="col1", index="col_group", aggfunc=[np.mean, np.median])
df.pivot_table(values="col1", index="col_group1", columns="col_group2") # group by two columns
df.pivot_table(values="col1", index="col_group1", columns="col_group2", fill_value=0) # group by two columns, fill missing values
df.pivot_table(values="col1", index="col_group1", columns="col_group2", fill_value=0, margins=True) # group by two columns, add row and column totals

## ROW INDEXES

# Set column as row INDEX
df = df.set_index("col")
# Reset data frame index
df.reset_index(inplace=True) # turns index as column
df.reset_index(drop=True, inplace=True) # discard index column
# Row indexes are useful because they allow you to subset much more easily. Instead of writing
df[df["col"].isin(["val1", "val2"])]
# you can write
df.loc[["val1", "val2"]]
# You can use multiple columns as index / multi-level/hierarchichal index
df = df.set_index(["col1", "col2"])
# Subsetting is done slightly differently
df.loc[[("val1_col1", "val1_col2"), ("val2_col1", "val2_col2")]]
# Sort by index
df.sort_index(level=["ind1", "ind2"], ascending=True)

# Slicing a data frame by index
    # first sort index
df.loc["val0":"val11"] # last value is included (this is different from list subsetting)
    # with multiple indexes
df.loc[("val1_ind1","val1_ind2"):("val2_ind1","val11_ind2")]
# This is particularly useful when indexes are dates. For example, you can pass a year as an argument without specifying month or day
df.loc["2013":"2015"]
df.loc["2013-01":"2015-12"]

# Slicing a data frame by column/s
df.loc[:, "col1":"col3"]

# Subsetting a data frame by row/column number / position
df.iloc[1:3, 6:8]

# Access the components of a date
df["col1"].dt.month
df["col1"].dt.year
# For example, if you have a column with a date, you can create a new column with the year
df["year"] = df["date"].dt.year

# Get all missing values of df == True
df.isna()

# Check if there are any missing values in each column
df.isna().any()

# Count missing values of columns
df.isna().sum()
df.isnull().sum()
df.isna().sum().plot(kind="bar") # plot nº of NaNs in a bar chart

# Remove rows with missing values
df.dropna()
df.dropna(subset = ['col1', 'col2']) # remove rows with missing values on specific columns

# Replace missing values with another value
df.fillna("MISSING")
df.col.fillna(0).astype('int') # replace hte missing values of a column with a 0 and convert to numerical

# Select the row/s with the smallest value in a given column
df.nsmallest(1,columns='year') # return one row with the smallest year

# Replacing the values of a column
df['col'] = df['col'].replace({'wrong_value':'right_value'})

# Replace the values of a column with missing values/NaNs
df['col'] = df['col'].replace([value1, value2], np.nan) # or
df['col'].replace([value1, value2], np.nan, inplace=True)

# Split a column with strings and create a new column for each part
df['col'].str.split("-", expand=True)

# Drop a column / delete a variable from a data frame
df.drop('col', axis='columns', inplace=True)

# For referring to columns both brackets and dot notation can be used
df['col']
df.col
# However, to create or overwrite a column, the left hand side always has to be in brackets notation

# Plot a column of a data frame in the Y axis, and the data frame's index in the X axis / Line plot with Pandas
import matplotlib as plt
df.col.plot()
plt.xlabel('IndexCol')
plt.ylabel('Col')
plt.title('Title')
plt.show()
# When you have a df with one index and two columns, the method plots the two lines in the same graph
# To plot multiple graphs
to_plot = pd.concat(df.col1, df.col2, axis=1)
to_plot.plot(subplots=True)
plt.show()
# For a BAR PLOT
to_plot.plot(kind='bar')
plt.show()
# Stacked bar plot
to_plot.plot(kind='bar', stacked=True)
plt.show()
# Order the bars
to_plot.sort_values().plot(kind='bar')
plt.show()
# Horizontal bars
to_plot.sort_values().plot(kind='barh')
plt.show()
# Box plot
df[['col1', 'col2']].plot(kind='box')
plt.show()
# Histogram
df.col1.plot(kind='hist')
plt.show()

# Stacked bar plot of proportions
props = df.groupby('group_col')['binary_col'].value_counts(normalize=True)
wide_props = props.unstack()
wide_props.plot(kind='bar', stacked=True)

# Create a datetime index for your data frame
# Combine / paste two columns with strings
df['datetimecol'] = pd.to_datetime(df.datecol.str.cat(df.timecol, sep=' '))
# Set it as index
df.set_index('datetimecol', inplace=True)

# Check whether a string is present in each row of a column
df.col.str.contains('pattern', na=False) # 'na=False' returns False when it finds a missing value

# Create a FREQUENCY TABLE between two variables (a tally of how many times each combination of values occurs)
pd.crosstab(df.col1, df.col2)

# Mapping one set of values to another
df['new_col'] = df.col.map({'old_value1':'new_value1', 'old_value2':'new_value2'})

# Change column type to categorical
cats = pd.CategoricalDtype(['short', 'medium', 'long'], ordered=True) # ordered categories
df['col'] = df.col.astype(cats)

# Add a new column to a data frame
df = pd.DataFrame({'temp_c': [17.0, 25.0]}, index=['Portland', 'Berkeley'])
df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32) # or
df.assign(temp_f=df['temp_c'] * 9 / 5 + 32)

# Calculate the proportion of a boolean variable
late_prop_samp = (late_shipments['col']=="value").mean()

# Check whether all conditions are true
(df['col']<1).all()


# %% JOINING DATA - MUTATING JOINS - PANDAS

## INNER JOIN: return rows with matching values in both tables
df.merge(df2, on="col", suffixes=("_df1", "_df2"))

## LEFT JOIN
df.merge(df2, on='col', how='left') # default is 'inner'

## RIGHT JOIN
df.merge(df2, on='col', how='right') # default is 'inner'

## OUTER JOIN
df.merge(df2, on='col', how='outer') # default is 'inner'
# it can be used to find rows that do not have a match

# Merge according to columns with differnt names
df.merge(df2, left_on='col1', right_on='col1_right')

# Add a column that specifies the result of the merge for each row
df1.merge(df2, on="id", indicator=True)

## MERGE A TABLE TO ITSELF: you can do this with any type of merge

# Merge on data frame indexes> the sintax is the same as before (on='id') except when keys do not have the same name, where you have to add
df.merge(df2, on='id', left_on='col_df', right_on='right_col', left_index=True, right_index=True) # you need to set this last two to True

# Read csv and set column as index
pd.read_csv('file.csv',  index_col='col')

# Verifying merges (if not valid, returns error message)
df.merge(df2, on='id', validate='one_to_one') # default is "none"
df.merge(df2, on='id', validate='one_to_many')
df.merge(df2, on='id', validate='many_to_many')
df.merge(df2, on='id', validate='many_to_one')

# Merge ORDERED data or TIME SERIES data

# MERGE ORDERED
pd.merge_ordered(df1, df1, on='date') # default of how argument is 'outer'
# You can INTERPOLATE missing data 
pd.merge_ordered(df1, df1, on='date', fill_method='ffill') # forward fill: with the last value
# When using merge_ordered() to merge on multiple columns, the order is important when you combine it with the forward fill feature. The function sorts the merge on columns in the order provided.

# MERGE AS OF (also very useful for TIME SERIES data)
# similar to a left merge_ordered, but matches on the nearest key column and not on exact matches
pd.merge_asof(df1, df2, on='date') # default 'direction' argument is 'backwards': assigns the last row where the key column value in the right table is less than or equal to the key column value in the left table
pd.merge_asof(df1, df2, on='date', direction='forward') # assings the last row in the right table where the key is equal or greater than the one in the left
pd.merge_asof(df1, df2, on='date', direction='nearest')


# %% JOINING DATA - FILTERING JOINS - PANDAS

# Filter observations from one table based on whether or not they match an observation in another table

# SEMI JOIN:
# - Returns observations in the left table that have a match in the right table.
# Only the columns from the left table are shown. 
# No duplicate rows are returned, even if there is a one-to-many relationship
df_3 = df1.merge(df2, on="id") # first do an inner join
df1["id"].isin(df_3["id"])

# ANTI JOIN:
# - Returns observations in the left table that DO NOT have a match in the right table.
# - Only the columns from the left table are shown. 
df_3 = df1.merge(df2, on="id", how="left", indicator=True) # first do an inner join
df_3.loc[df_3["_merge"] == 'left_only', 'gid']


# %% JOINING DATA - CONCATENATION - PANDAS

# Vertical bind / row bind (default)
pd.concat([df1, df2, df3], axis=0) # or 
pd.concat([df1, df2, df3], axis='index') 
pd.concat([df1, df2, df3], axis=1) # or
pd.concat([df1, df2, df3], axis='columns')
    # You can ignore the index
pd.concat([df1, df2, df3], ignore_index=True)
    # Add keys labels to identify which row came from which data frame
pd.concat([df1, df2, df3], ignore_index=False, keys=["1", "2", "3"])
    # You can bind two tables where one has more rows than the other. The method will keep all columns. You can sort columns alphabetically
pd.concat([df1, df2, df3], sort=True)
    # Only keep matching columns
pd.concat([df1, df2, df3], join='inner') # default is 'outer'

# Append method on data frames: simplified version of concat
df.append([df2, df3], ignore_index=True, sort=True)

# Horizontal bind / row bind (default)
pd.concat([df1, df2, df3])

# Verifying concatenations
df.concat(verify_integrity=True) # deafult is False. True verifies if there are duplicated indexes

# %% SELECTING DATA - QUERY 
# Similar to the WHERE clause of a SQL statement
df.query('col > 10') # returns all rows where col is grater than 10
df.query('col > 10 and col < 20')
df.query('col > 10 or col2 == "value"') # use double quotes inside the statement

# %% RESHAPING DATA - melt

# Reshape data from wide to long format
df.melt(id_vars=['col1', 'col2'])
    # chose which variables will remain unpivoted
df.melt(id_vars=['col1', 'col2'], value_vars=['2019', '2020'])
    # Set names for the new variable column and the values column
df.melt(id_vars=['col1', 'col2'], var_name='years', value_name='dollars')


# %% DATA VISUALIZATION WITH MATPLOTLIB

# Gallery: https://matplotlib.org/2.0.2/gallery.html
# Working with images: https://matplotlib.org/stable/tutorials/introductory/images.html
# Animations: https://matplotlib.org/stable/api/animation_api.html
# Geospatial data: https://scitools.org.uk/cartopy/docs/latest/

import matplotlib.pyplot as plt

x=[1,2,3,4]

y=[4,65,12,4]

z=[3,3,2,1]

# To visualize graph, show it
plt.show()

# To close visualization
plt.clf()

# Line plot
plt.plot(x, y, kind="line") # here "line" is default, so not necessary

# Scater plot
plt.scatter(x, y) # or df.plot(x,y,kind="scatter")
    # Set dot size according to a variable
plt.scatter(x, y, s=z)

# Histogram
plt.hist(x)
df['col'].hist() # with pandas
plt.hist(df.dropna(), bins=30)
# control bin number
plt.hist(bins=29)
plt.hist(bins=np.arange(10, 130, 20)) # from 10 to 110 with binwidth of 20 (the top value is exclusive)

# Title
plt.title("Title")

# X and Y labels
plt.xlabel()
plt.ylabel()

# Change to logarithmic scale
plt.scale("log")
plt.yscale('log')
plt.xscale('log')

# Bar plot
plt.plot(kind="bar")
df.plot(kind="bar") # pandas

# Place two or more graphs in the same plot
df.plt.hist(alpha=0.7)
df2.plt.hist(alpha=0.7)
plt.show()

# Plot multiplie histogramas at a time, but not on the same plot
df.[["col1", "col2"]].hist()

# Add legend to plot
plt.legend(["A", "B"])


# (INTRODUCTION TO DATA VISUALIZATION WITH MATPLOTLIB)

# Create figure and axis objects
# Figure: container that holds everything you see on the page
# Axis: part of the page that holds the data
fig, ax = plt.subplots()
plt.show()

# LINE CHARTS

# Plotting command: methods of the axis object
ax.plot(df['col1'], df['col2'])
plt.show()

# Add more data (multiple line plots in the same figure)
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'])
ax.plot(df2['col1'], df2['col2'])
plt.show()

# Line plot with dots (markers)
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'], marker='o') # other options: https://matplotlib.org/stable/api/markers_api.html
plt.show()
# Set marker size
ax.plot(df['col1'], df['col2'], marker='o', markersize=2)

# Change the linestyle of a line plot
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'], linestyle="--") # other options: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
plt.show() # another option is "None"

# Change color of line graph
ax.plot(df['col1'], df['col2'], color='r') # Red

# Customize axis labels
ax.set_xlabel('Time')
ax.set_ylabel('Time')

# Title
ax.set_title('Title')

# Multiple plots in the same figure ('Small multiples': similar data across diferent groups/subjects)
fig, ax = plt.subplots(3, 2) # figure object with 3 rows, 2 columns
plt.show()
# Now "ax" is an array of axis objects
# See dimensions of axis
ax.shape
# Add a plot to one of the axis objects
ax[0,0].plot(df['col1'], df['col2']) # top left subplot
# only one column
fig, ax = plt.subplots(3, ) # figure object with 3 rows, 1 columns
plt.show()
ax[0].plot(df['col1'], df['col2']) # top subplot

# Set equal range of axis in subplots
fig, ax = plt.subplots(2, 1, sharey=True)

# Plot graphs from two different data frames in the same chart
ax = df1.plot(x='time', y='value', label='DF 1')
df2.plot(x='time', y='value', label='DF 2', ax=ax, ylabel="Y title")
plt.show()

# Plotting TIME SERIES data
# First set date column as index of pandas data frame
df = pd.read_csv('df.csv', parse_dates=['date'], index_col='date')
# Add the index as the X axis
ax.plot(df.index, df['col'])

# Use two Y axis scales
fig, ax = plt.subplots()
ax.plot(df.index, df['col1'], color='blue')
ax.set_ylabel('Axis 1', color='blue') # color may also be passed here
ax.tick_params('y', colors='blue') # you can also set color of ticks and ticks label 
ax2 = ax.twinx() # share the same x axis, but not the same y axis
ax2.plot(df.index, df['col2'], color='red')
ax2.set_ylabel('Axis 2', color='red')
ax2.tick_params('y', colors='red') # you can also set color of ticks and ticks label 
plt.show()

# Create a function to plot a time series
def plot_timeseries(axes, x, y color, xlabel, ylabel):
    axes.plot(x, y, color=color)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.tick_params('y', colors=color)
# Replicating the exercise above
fig, ax = plt.subplots()
plot_timeseries(ax, df.index, df['col1'], 'blue', 'Xlab', 'Ylab')
plot_timeseries(ax, df.index, df['col2'], 'red', 'Xlab2', 'Ylab2')
plt.show()

# Add annotations (to a time series plot)
ax.annotate("Annotation", xy = (pd.Timestamp('2015-10-06'), 1), \ # xy is the position of the point
    xytext = (pd.Timestamp('2015-10-06'), -0.2), \# xytext is the position of the text
    arrowprops={"arrowstyle":"->", "color":"gray"}) # "Arrow properties": Add an arrow that points from the text to the data point
# More options at: https://matplotlib.org/stable/tutorials/text/annotations.html

# Annotate a scatter plot with point labels
for x, y, label in zip(xs, ys, labels):
    plt.annotate(label, (x, y), fontsize=5, alpha=0.75)
plt.show()


# BAR CHARTS

# Create a bar chart
fig, ax = plt.sbuplots()
ax.bar(df.index, df['col'])
plt.show()

# Rotate tick labels
ax.set_xticklabels(df.index, rotation=90)

# Stacked bar chart
fig, ax = plt.sbuplots()
ax.bar(df.index, df['col1'])
ax.bar(df.index, df['col2'], bottom=df['col1']) # the bottom of col2 bars should be the height of col1 bars
ax.bar(df.index, df['col3'], bottom=df['col1']+df['col1'])
plt.show()

# Add a legend to stackedd bar chart
fig, ax = plt.sbuplots()
ax.bar(df.index, df['col1'], label="COL 1")
ax.bar(df.index, df['col2'], bottom=df['col1'], label="COL 2") # first, define labels
ax.bar(df.index, df['col3'], bottom=df['col1']+df['col1'], label="COL 3")
ax.legend() # call axes legend method
plt.show()

# HISTOGRAMS

# Multiple histograms in the same plot
fig, ax = subplots()
ax.hist(df['col1'], label="C1")
ax.hist(df['col2'], label="C2")
ax.legend()
plt.show()

# Set number of bins of a histogram. Two options
    # Quantity of bins
ax.hist(df.col, bins=5)
    # Sequence of values
ax.hist(df.col, bins=[10, 20, 30, 40, 50])

# Change histogram type
ax.hist(df.col, bins=5, histtype="step") # displays the histogram with thin lines instead of solid bars

# STATISTICAL PLOTTING

# Add error bars to bar charts
fig, ax = subplots()
ax.bar("group1", df['g1'].mean(), yerr=df['g1'].std())

# Add error bars to line charts
fig, ax = subplots()
ax.errorbar(df['col1'], df['col2'], yerr=df['col2_std'])

# BOXPLOTS
# Create two boxplots in the same plot
fig, ax = subplots()
ax.boxplot([df['col1'], df['col2']])
ax.set_xticklabels(['Col1', 'Col2'])
plt.show()

# SCATTER PLOTS
fig, ax = subplots()
ax.scatter(df['col1'], df['col2'])
plt.show()

# Create scatter plot with multiple groups of points
# eg with time series
sixties = df["1960-01-01": "1960-12-31"]
seventies = df["1970-01-01": "1970-12-31"]
fig, ax = subplots()
ax.scatter(sixties['col1'], sixties['col2'], color='blue', label='sixties')
ax.scatter(seventies['col1'], seventies['col2'], color='red', label='seventies')
ax.legend()
plt.show()

# Encoding a third variable by color (as a gradient of color)
fig, ax = subplots()
ax.scatter(df['col1'], df['col2'], c=df.index) # here index could be a time series index

# PREPARING YOUR FIGURES TO SHARE WITH OTHERS

# Changing the overall style of the figure / Set figure style
plt.style.use("ggplot") # This style will apply to all figures in the session until you change it to another style
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'])
ax.plot(df2['col1'], df2['col2'])
plt.show()
# Go back to default style
plt.style.use("default") 
# See all styles: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
# Guidelines:
    # - Avoid dark backgrounds
    # - Preer colorblind options ("seaborn-colorblind", "tableau-colorblind10")
    # If printed with color, use less ink
    # f printed black and white, use "grayscale"

# SAVING FIGURES

# Replace plt.show() with a call to fig.savefig()
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'])
fig.savefig("file.png", quality=50) # Also "jpg", "svg". Avid quality values above 95
# Or control dpi (dots per inch, resolution)
fig.savefig("file.png", dpi=300)

# Control figure size
fig, ax = plt.subplots()
ax.plot(df['col1'], df['col2'])
fig.set_size_inches([4,5]) # Height and width
fig.savefig("file.png", dpi=300)

# JOINTPLOTS: scatter plots together with histograms of both variables
sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'])

# Adding %-formatting to the y-axis
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# Zoom into an area of a plot / set axis limits
plt.plot(x, y, 'o', markersize=1, alpha=0.02)
plt.axis([140, 200, 0, 160])
plt.show()

# Set equal scale for both axes
plt.axis('equal')

# %% INTRODUCTION TO DATA VISUALIZATION WITH SEABORN
# https://seaborn.pydata.org/

# Built in dataset
tips = sns.load_dataset("tips")

# Built on matplotlib, used to work with pandas data frames.

import seaborn as sns
import matplotlib.pyplot as plt # you also need to import matplotlib

# Make a plot with seaborn
sports = ['football', 'baseball', 'baseball', 'baseball', 'baseball', 'football']
sns.countplot(x=sports) # or sns.countplot(y=sports)
plt.show()

# Using pandas with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Scatter plot (in Seaborn "relplots" are prefered to scatterplots)
sns.scatterplot(x='col1', y='col2', data=df)
plt.show()

# Add a third variable as hue (color) (color by group)
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips,
    hue="smoker", # variable for hue (legend is added automatically)
    hue_order=["Yes", "No"], # order of labels in legend
    palette={"Yes":"black", "No":"red"} # define palette of colors
    ) 
plt.show()

# RELATIONAL PLOTS

# Creating multiple plots in figure (relplot())
# Relplot - SCATTERPLOTS
tips = sns.load_dataset("tips")
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", col="smoker") # use the col argument
plt.show()
# Arrange multiple plots vertically
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", row="smoker") # or row argument
# Use two dimensions
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", col="smoker", row="time")
# Set number of plots per row
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", col="day", col_wrap=2)
# Change the order of the plots
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", col="day", col_wrap=2, col_order=["Thur", "Fri", "Sat", "Sun"])

# Customizing scatterplots (can be used in both scatterplots and relplots)
tips = sns.load_dataset("tips")
    # Varying the point size accoridng to a column
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", size="size")
    # Add color variation according to point size
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", size="size", hue="size") # size is quantitative variable
    # Varying point style
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", hue="smoker", style="smoker")
    # Varying point transparency
sns.relplot(x="tota_bill", y="tip", data=tips, kind="scatter", alpha=0.4)

# LINE PLOTS
import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(x='col1', y='col2', data=df, kind="line")
plt.show()
# Or
sns.lineplot(data=tips, x='total_bill', y='tip')

# Add a trendline to a scatter plot
sns.lmplot(x='col1', y='col2', data=df, ci=None)

# Multiple line plots with different colors and line styles
sns.relplot(x='col1', y='col2', data=df, kind="line", hue="col3", style="col3")
# if you dont want the line style to vary by group, remove style argument and add argument dashes=False
# If a line plot is fed multiple observations for each X-value, it aggregates them and shows the mean and a 95% confidence interval for the mean
# If instead of a 95% confidence interval you want 1 standard deviation, use
sns.relplot(x='col1', y='col2', data=df, kind="line", ci="sd")
# Or turn it off
sns.relplot(x='col1', y='col2', data=df, kind="line", ci=None)

# Different point markers
sns.relplot(x='col1', y='col2', data=df, kind="line", hue="col3", style="col3", markers=True)

# CATEGORICAL PLOTS: COUNT PLOT, BAR PLOT and POINT PLOT
sns.catplot(x='col1', data=df, kind="count") # count plot
sns.catplot(x='col1', y='col2', data=df, kind="bar") # bar plot (shows mean of variable with CIs)

# Order the bars of a categorical plot
cat_order = ['cat1', 'cat2']
sns.catplot(x='col1', data=df, kind="count", order=cat_order)

# Change the orientation of the bars in a categorical plot
sns.catplot(x='col2', y='col1', data=df, kind="bar") # just switch the X and Y variables
sns.catplot(y='col1', data=df, kind="count", order=cat_order)

# BOX PLOTS
sns.catplot(x='var_grupo', y='col1', data=df, kind='box')

# Change the order of the boxplots
cat_order = ['cat1', 'cat2']
sns.catplot(x='var_grupo', y='col1', data=df, kind='box', order=cat_order)

# Omit outliers
sns.catplot(x='var_grupo', y='col1', data=df, kind='box', sym='') # sym can also be changed to change te appearence of the outliers

# Change the whiskers
sns.catplot(x='var_grupo', y='col1', data=df, kind='box', whis=2.0) # Extend to 2x interquantile range
sns.catplot(x='var_grupo', y='col1', data=df, kind='box', whis=[5,95]) # Extend to a specific percentile
sns.catplot(x='var_grupo', y='col1', data=df, kind='box', whis=[0,100]) # Extend to min and max values

# POINT PLOTS: show the mean of a quantitative variable for the mean of observations in each category, with 95% CI
sns.catplot(x='cat_var', y='var1', data=df, kind='point')
sns.catplot(x='cat_var', y='var1', data=df, kind='point', hue='var2') # add multiple lines with diff colors
sns.catplot(x='cat_var', y='var1', data=df, kind='point', join=False) # do not join points
# Us median instead of mean
from numpy import median
sns.catplot(x='cat_var', y='var1', data=df, kind='point', estimator=median)
# Add caps to the confidence intervals
sns.catplot(x='cat_var', y='var1', data=df, kind='point', capsize=0.2)

# Pre-set FIGURE STYLES: white (default), dark, whitegrid, darkgrid, ticks
# Set figure style
sns.set_style("whitegrid")
# 'whitegrid'  Add a grid in the background
# 'ticks'  Add tick marks
# 'dark'  Add a grey background
# 'dark'  Add a grey background and white grid

# Pre-set seaborn DIVERGING PALETTES: "RdBu", "PRGn"
# Set color palette
sns.set_palette("RdBu")
sns.catplot(x='col1', data=df, kind="count")
# Reverse a palette: "RdBu_r", "PRGn_r"

# Pre-set seaborn SEQUENTIAL PALETTES: Greys, Blues, PuRd, GnBu (best for continuous variable)
sns.set_palette("Blues")

# Create custom palette
custom_palette = ['red', 'green', 'orange', 'blue', 'yellow', 'purple']
sns.set_palette(custom_palette)
custom_palette = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC'] # with hex color codes
sns.set_palette(custom_palette)

# Change the SCALE STYLE / SIZE of a plot: paper (default), notebook, talk, poster (basically, the size of the labels relative to the plot increases as you go right in the options)
sns.set_context('notebook')

# PLOT TITLES AND AXIS LABELS
# Seaborn plots create two different types of objects: FacetGrid and AxesSubplot
# How to see this:
g = sns.scatterplot(x='col1', y='col2', data=df)
type(g) # returns "matplotlib.axes._subplots.AxesSubplot"
# While
g = sns.catplot(x='col1', y='col2', data=df, kind='box')
type(g) # returns 'seaborn.axisgrid.FacetGrid'
# a FacetGrid consists of one or more axes subplots. relplot() and catplot() can create subplots
# AxesSubplots, on the other hand, can only contain one plot. Like scatterplot() or countplot(), etc, which return a single AxesSubplot object

# Add a TITLE to a FacetGrid object
g = sns.catplot(x='col1', y='col2', data=df, kind='box') # First assign the plot to an object
g.fig.suptitle('New title') # now use subtitle method
# Adjust the height of the title
g.fig.suptitle('New title', y=1.05) 

# Add a TITLE to an AxesSubplot object
g = sns.scatterplot(x='col1', y='col2', data=df)
g.set_title('New title', y=1.05)
# Set it for various subplots
g = sns.catplot(x='col1', y='col2', data=df, kind='box', col='var3')
g.set_titles('This is {col_name}', y=1.05)

# Add AXIS LABELS
g = sns.catplot(x='col1', y='col2', data=df, kind='box', col='var3')
g.set(xlabel= 'xlab',
    ylabel='ylab')

# ROTATE tick labels
g = sns.catplot(x='col1', y='col2', data=df, kind='box', col='var3')
plt.xticks(rotation=90)


# %% INTERMEDIATE DATA VISUALIZATION WITH SEABORN
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

# HISTOGRAM
sns.histplot(tips['total_bill'])

# More general: DISTRIBUTION PLOT (best plot to start with)
sns.displot(tips['total_bill']) # default is histogram
# KERNEL DENSITY PLOT
sns.displot(tips['total_bill'], kind='kde')
# Set number of bins, add kernel density plot
sns.displot(tips['total_bill'], bins=10, kde=True)
# Add tick marks for each observation
sns.displot(tips['total_bill'], rug=True)
sns.displot(tips['total_bill'], kind='kde', rug=True, fill=True)
# Estimated cumulative density function
sns.displot(tips['total_bill'], kind='ecdf') 

# REGRESSION PLOTS
sns.regplot(data=tips, x='total_bill', y='tip')

# FACETING: the use of plotting multiple graphs by changing only a single variable

# LM PLOT: relation between 2 variables
sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex') # regression line for each value of "hue" variable
sns.lmplot(data=tips, x='total_bill', y='tip', col='sex') # one graph for each value of "hue" variable
sns.lmplot(data=tips, x='total_bill', y='tip', row='sex')

# SEABORN STYLES
# Compare styles
for style in ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']:
    sns.set_style(style)
    sns.displot(tips['total_bill'])
    plt.show()

# Remove axis lines
sns.histplot(tips['total_bill'])
sns.despine(left=True) # left Y axis. Also top, right, bottom

# COLORS

# Use matplotlib color codes
sns.set(color_codes = True)
sns.displot(tips['total_bill'], color= 'g')

# Define PALETTE of colors
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
for p in palettes:
    sns.set_palette(p)
    sns.palplot(sns.color_palette()) # Display a color palette
    plt.show()
# You can also define it within a plot
sns.violinplot(data=df,
         x='Award_Amount',
         y='Model Selected',
         palette='husl')

# Defining custom color palettes
# Circular colors: when data is not ordered
sns.palplot(sns.color_palette("Paired", 12)) # 12 colors. palplot is to plot the palette
# Sequential colors: when data has a consistent range from high to low
sns.palplot(sns.color_palette("Blues", 12))
# Diverging colors: when both the low and high values are interesting
sns.palplot(sns.color_palette("BrBG", 12))

# Customizing with matplotlib
# Set a limit to the axis scale
fig, ax = plt.subplots()
sns.histplot(tips['total_bill'], ax=ax)
ax.set(xlim = (0,40)) # you can either 

# Set figure size
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,4))
sns.histplot(tips['total_bill'], ax=ax0)
sns.histplot(tips['size'], ax=ax1)
# Graph a vertical line
ax0.axvline(x=20, label='line', linestyle='--', linewidth=2)
ax0.legend()

## CATEGORICAL PLOTS
# 3 GROUPS:
# 1) The ones that show all individual observations on the plot: STRIPPLOT and SWARMPLOT
# 2) Abstract representations: BOXPLOT, VIOLIN PLOT, BOXENPLOT
# 3) Statistical estimates: BARPLOT, POINTPLOT, COUNTPLOT
tips = sns.load_dataset("tips")

# STRIPPLOT
sns.stripplot(data=tips, y='total_bill', x='sex')
sns.stripplot(data=tips, y='total_bill', x='sex', jitter=True) # jitter adds a random /noise (like wiggle) sideways so that points are more easily visualized
# This is particularly appropiate when the variable is rounded of to the nearest integer number, so that without doing this the discrete values are less intuitive to visualize

# Jitter two variables manually
import numpy as np
x_jitter = x + np.random.random(0, 2, size=len(df))
y_jitter = y + np.random.random(0, 2, size=len(df))
plt.plot(x, y, 'o', markersize=1, alpha=0.01)
plt.show()

# SWARMPLOT: points will not overlap, but it is not so convenient for very large data sets
sns.swarmplot(data=tips, y='total_bill', x='sex') 

# VIOLIN PLOT: combination of a kernel density plot and a boxplot (takes time to render)
sns.violinplot(data=tips, y='total_bill', x='sex')
sns.violinplot(data=tips, y='total_bill', x='sex', inner=None) # innter: Representation of the datapoints in the violin interior.

# BOXENPLOT: enhanced box plot (quicker than a violin plot)
sns.boxenplot(data=tips, y='total_bill', x='sex')

# BARPLOT
sns.barplot(data=tips, y='total_bill', x='smoker', hue='sex')

# POINTPLOT
sns.pointplot(data=tips, y='total_bill', x='smoker', hue='sex')


## REGRESSION PLOTS

# REGPLOT
sns.regplot(data=tips, y='total_bill', x='tip') # linear regression line
# Add a marker to each point
sns.regplot(data=tips, y='total_bill', x='tip', marker='+')
# Polinomial regression
sns.regplot(data=tips, y='total_bill', x='tip', order=2) # of order 2
# Add an estimator
sns.regplot(data=tips, y='total_bill', x='tip', x_estimator=np.mean) # !!! fix example
# Disable the regression line (just the scatter plot)
sns.regplot(data=tips, y='total_bill', x='tip', fit_reg=False)

# RESIDUAL PLOT: plots the residuals of regression between y and x
sns.residplot(data=tips, y='total_bill', x='tip') # linear regression
sns.residplot(data=tips, y='total_bill', x='tip', order=2) # polynomial of order 2

## MATRIX PLOTS

# Create a matrix-like object from two variables of a data frame / CROSS-TABULATION
mat = pd.crosstab(tips['day'], tips['time'], values=tips['total_bill'], aggfunc='mean')

# HEATMAP
sns.heatmap(mat)
# Turn on annotations in individual cells / display cell value in numbers
sns.heatmap(mat, annot=True)
# Define a color map
sns.heatmap(mat, annot=True, cmap="YlGnBu")
# Hide the color bar
sns.heatmap(mat, annot=True, cmap="YlGnBu", cbar=False)
# Define the width of the lines between the cells
sns.heatmap(mat, annot=True, cmap="YlGnBu", cbar=False, linewidths=.5)
# Center the color scheme on a psecific value 
sns.heatmap(mat, annot=True, cmap="YlGnBu", cbar=False, linewidths=.5, center=mat.loc[1,2])

# CORRELATION MATRIX PLOT
sns.heatmap(tips[list(tips.columns)].corr(), cmap='YlGnBu')

# FACETTING: eg. multiple plots of the same variable across categories
g = sns.FacetGrid(tips, col='sex')
g.map(sns.boxplot, 'total_bill')
# catplot() is a shortcut to create a facet grid more easily
sns.catplot(x='total_bill', data=tips, col='sex', kind='box')
# Facet grid accept matplootlib plots
import matplotlib.pyplot as plt
g = sns.FacetGrid(tips, col='sex')
g.map(plt.scatter, 'total_bill', 'tip')
# lmplot() plots scatter and regression plots on a FacetGrid
sns.lmplot(data=tips, x='total_bill', y='tip', col='sex', col_order=['Male', 'Female'], row='day', fit_reg=False)

# PAIRGRID: shows pairwise relationships between data elements
g = sns.PairGrid(tips, vars=['total_bill', 'tip'])
g.map(sns.scatterplot)
# Define graphs on diagonals and off diagonls
g = sns.PairGrid(tips, vars=['total_bill', 'tip'])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
# pairplot is a shortcut function to a pair grid
sns.pairplot(tips, vars=['total_bill', 'tip'], kind='reg', diag_kind='hist', hue='sex', palette='husl')
# Increase transparency
sns.pairplot(tips, vars=['total_bill', 'tip'], diag_kind='hist', hue='sex', palette='husl', plot_kws={'alpha':0.5})

# JOINT GRID: compare distribution between two variables
g = sns.JointGrid(tips, x='total_bill', y='tip')
g.plot(sns.regplot, sns.histplot)
# Advanced jointgrid: bivariate kernel density plot
g = sns.JointGrid(tips, x='total_bill', y='tip')
g.plot_joint(sns.kdeplot)
g.plot_marginals(sns.kdeplot, fill=True)
# Shortcut with jointplot()
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
# Overlay a kernel density plot over the scatter plot
sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter').plot_joint(sns.kdeplot)

# %% INTRODUCTION TO STATISTICS

# Absolute value of a number
import numpy as np
np.abs(-22)

# Calculate the mode
import statistics
statistics.mode(df['col'])

# Variance
np.var(df['col']) # population variance
np.var(df['col'], ddof=1) #  sample variance, default is 0

# Standar deviation
np.std(df['col']) # population
np.std(df['col'], ddof=1) #  sample

# Mean absolute deviation (like variance but without squaring. It penalizes distances equally)
np.mean(np.abs(df['col']-mean(df$col)))

# Quantiles
np.quantile(df['col'], 0.5) # median
np.quantile(df['col'], [0.5,0.9])
np.quantile(df['col'], np.linspace(0,1,5)) # five quantiles with equal distance

# Interquantile range
from scipy.statshist import iqr
iqr(df['col'])

# Outliers according to boxplot criterion
from scipy.stats import iqr
iqr(df['col'])
lower_threshold = np.quantile

# Sample a data frame
df.sample(n=10) # without replacement
df.sample(n=10, replace=True) # with replacement

# Calculate cumulative probabilities (less than or equal to) of a Uniform distribution
from scipy.stats import uniform
uniform.cdf(7, 0, 12) # pointvalue=7, lower=0, upper=12. Prob of getting 7 in a uniform range of 0-12

# Generate random numbers with a uniform distribution
from scipy.stats import uniform
uniform.rvs(0, 5, size=10) # min, max, n

# Generate a variable with binomial distribution
from scipy.stats import uniform
binom.rvs(1, 0.5, size=100) # n of coins, prob of success, n of trials

# Calculate point probabilities of a Binomial distribution
from scipy.stats import binom
binom.pmf(5, 100, 0.5) #n of success, n trials, prop of success

# Calculate cumulative probabilities (less than or equal to) of a Binomial distribution
from scipy.stats import binom
binom.cdf(5, 100, 0.5) #n of success, n trials, prop of success

# Calculate cumulative probabilities (less than or equal to) of a normal distribution
from scipy.stats import norm
norm.cdf(50, 47, 5) # value, mean, sd

# Calculate point values of percentiles of a normal distribution
from scipy.stats import norm
norm.ppf(0.9, 47, 5) # value, mean, sd

# Generate random numbers with a normal distribution
norm.rvs(0, 1, size=100) # mean, sd, n

# Calculate point probabilities of a Poisson distribution
from scipy.stats import poisson
poisson.pmf(5, 8) # value, lambda (mean rate)

# Calculate cumulative probabilities (less than or equal to) of a Poisson distribution
from scipy.stats import norm
poisson.cdf(5, 8) # value, lambda (mean rate)

# Generate random numbers with a Poisson distribution
norm.rvs(8, size=100) # lambda, size

# Calculate cumulative probabilities (less than or equal to) of a Exponential distribution
from scipy.stats import expon
expon.cdf(5, 8) # value, 1/lambda (1/mean Poisson rate)

# Calculate the (Pearson) correlation between two variables
df['col1'].corr(df['col2'])
# Numpy version
from scipy.stats import pearsonr
correlation, pvalue = pearsonr(df['col1'], df['col2'])

# %% INTRODUCTION TO NUMPY
# It is a foundational library on which others like scikit-learn, scipy, matplotlib, pandas and tensorflow are built
# Array: a grid like object that holds data. Can have any number of dimensions, and each dimension can have any length. Admit only one data type, uses less memory.
import numpy as np

# Create an array from a list
list1=[1,5,1,2,3,6]
array=np.array(list1)
type(array)
# 2 dimensions
list_of_lists= [[1,4,3],
                [7,4,7],
                [10,5,2]]
np.array(list_of_lists)
# 3 dimensions: list of lists of lists

# Other ways of creating arrays
np.zeros((2,3)) # 2 rows, 3 columns of zeros
np.ramdom.ramdom((3,6)) # random numbers between 0 and 1 (function "random" of numpy module "random")
np.arange(-3,4) # evenly spaced array of consecutive integers, from -3 to 4 (excluded)
np.arange(4) # beings at 0
np.arange(-3, 4, 3) # steps of 3
np.array(range(1,13)) # Create a sequence of numbers

# Return the dimension of an array
array.shape

# Convert a mult-dimensional array into a one-dimensional array
array.flatten()

# Redefine the shape of an array
array.reshape((2,3)) # the tuple passed must be compatible with the number of elements of the original array

# NUMPY DATA TYPES: more specific than Python data types
# They include the type of data (integer, float, string) AND the available memory in bits
# Eg: 'np.int64', 'np.float32', '<U12' (unicode string with max length 12), 'bool_'

# Return the data type of an array
array.dtype

# You can determine the data type when creating an array
np.array([1.23,4.5,1], dtype=np.float32)

# Convert array data type (for example to reduce memory usage)
array.astype(np.int8)

# INDEXING arrays
array[0] # one dimension
array[2,4] # two dimensions. third row of fifth column
array[1] # second row
array[:, 3] # fourth column

# SLICING arrays
array[2:4] # one dimension (2 is included, 4 is not)
array[2:4, 5:8] # two dimensions
array[2:4:1, 5:8:1] # two dimensions, with step value (skip one in columns)

# SORTING an array along an axis
np.sort(array) # axis 1, by column (so each row is ordered from smallest to largest)
# The default axis is the last axis of the array passed to it. If the array is 2D, then the last axis is columns
np.sort(array, axis=0) # axis 0, by rows (so each column is ordered from smallest to largest)

# FILTERING
# Boolean mask: an array of booleans with the same shape as the filtered array
# Fancy indexing: returns an array of elements which meet a certain condition
one_to_five = np.array([1,2,3,4,5])
mask = one_to_five % 2 == 0
one_to_five[mask]
# ... with 2D arrays
classroom_ids_and_sizes = np.array([[1,22], [2,21], [3,27], [4,26]])
# eg check which values in second column are even
mask = classroom_ids_and_sizes[:, 1] % 2 == 0
# then index the first column using that mask, to return the class ids
classroom_ids_and_sizes[:, 0][mask]

# Filtering with NP.WHERE: returns an array of indices of elements that meet a certain condition
np.where(classroom_ids_and_sizes[:, 1] % 2 == 0) # returns a tupple of arrays, one for each dimension index
# For 2D arrays, it returns two tupples of arrays, because identifying each element requires two indices
row_index, col_index = np.where(sudoku==0) # it is helpful to unpack the indices in two objects

# Replace all elements of an array with a specific value
np.where(array == value, 'replace_with_this_if_condition_is_met', array) # change to that string, otherwise return original element

# ADDING AND REMOVING ELEMENTS OF AN ARRAY

# Concatenation (arrays must have compatible shapes and dimensions. it never adds new dimensions)
np.concatenate((array1, array2)) # along the first axis (default axis=0, adding new rows) 
np.concatenate((array1, array2), axis=1) # along the second axixs, adds new columns
# To concatenate a 2D array with a 1D array, first you need to reshape the 1D array
array_1D = np.array([1,2,3])
column_array_2D = array.1D.reshape((3,1)) # here you "add" the column
column_array_2D

# Delete row of array
np.delete(array, 1, axis=0) # in the position of 1 there can be a slice, index or array of indices. Here you  delete the second row.
# Delete column of array
np.delete(array, 3, axis=1)

# SUMMARIZING DATA
# AGREGATING METHODS
array.sum() # sums all elements in array
array.sum(axis=0) # sum all rows in each column (create column totals)
array.sum(axis=1) # sum all columns in each row (create row totals)
array.sum(axis=1, keepdims=True) # If True, the dimensions that are collapsed when aggregating are left in the output array and set to 1 (for dimension compatibility)
array.min()
array.max()
array.mean()
array.cumsum(axis=0)

# VECTORIZED OPERATIONS
# Use of optimized code in C language
array + 3 # add a scalar
array * 3 # multiply
array1 + array2 # add compatible arrays
array1 * array2
array > 3 # boolean mask

# Convert Python functions to a numpy vectorized function
array = np.array(["NumPy", "is", "awesome"]) # eg, check the length of each element
len(array) > 2 # this does not work element-wise because len is a Python function
vectorized_len = np.vectorize(len) # create vectorized function
vectorized_len(array) > 2

# BROADCASTING: math operations between arrays of different shapes
# Summing a scalar to an array is an example
# Still arrays have to be somewhat compatible
# Numpy compares sets of array dimensions from right to left
# Two dimensions are compatible when one of them has a length of one or they are of equal lengths
# Both arrays have to be compatible across all of their dimensions. But the two arrays do not need to have the same number of dimensions.
# Examples: (10,5) and (10, 1) / (10,5) and (5,) are compatible. (10,5) and (5,10) / (10,5) and (10,) are NOT compatible
np.arange(10.reshape((2,5))) + np.array([0,1,2,3,4])
# The assumption is that the user is trying to broadcast row-wise
# Eg this does not work
np.arange(10).reshape((2,5)) + np.array([0,1])
# But this does
np.arange(10).reshape((2,5)) + np.array([0,1]).reshape((2,1))

# SAVING AND LOADING ARRAYS

# RGB ARRAY: each value describes the red, green and blue component of a single picture
rgb = np.array([[[255,0,0], [255,0,0], [255,0,0]],
                [[0,255,0], [0,255,0], [0,255,0]],
                [[0,0,255], [0,0,255], [0,0,255]]
                ])
plt.imshow(rgb)
plt.show()

# Arays con be saved in .csv, .txt, .pkl, and .npy (ideal)

# Load a numpy array
with open("file.npy", "rb") as f: # Open takes 2 arguments: the name of the file and the open mode (here: "read binary")
    file_array = np.load(f)

# Save a numpy array
with open("file2.npy", "wb") as f: # wb: "write binary"
    np.save(f, file2)

# Call the documentation of a numpy method
help(np.ndarray.flatten)

# ARRAY ACROBATICS: changing axis order (useful for data augmentation, eg flipping images)

# Flip an array along every axis
np.flip(array)
np.flip(array, axis=(0,1)) # flip along specific axis

# Transpose an array
np.transpose(array)
np.transpose(array, axes=(1,0,2)) # specify transpose order (here makes column to rows but leaves 3rd dim untouched) (!!! its "axEs" not "axis". Order matters here)

# STACKING AND SPLITTING
# Split an array
red, green, blue = np.split(rgb_array, 3, axis=2) # array, number of equally sized arrays desired after the split, the axis to plit along (here: split rgb along the third axis into 3 arrays to isolte red green and blue values of an rgb image)
# Each object created will have the same number of dimensions as the original array

# Stack arrays
# Remember that it is not possible to concatenate data in a new dimension. Instead we use stacking
# All arrays must have the same shape and number of dimensions
np.stack([array1, array2], axis=2) # axis 



# %% PYTHON DATA SCIENCE TOOLBOX 1

# Define a function
def square(value):   # function header, parameter
    new_value= value**2 # function body
    return new_value
square(5) # now 5 is the "argument"

# DOCSTRINGS
# Describe what your function does, serves as documentation
def square(value):   # function header, parameter
    """ Return the square of a value."""
    new_value= value**2 # function body
    return new_value

# Return multiple values
def raise_both(value1, value2):
    """ Raise value1 to the power of value2 and vice versa."""
    new_value1 = value1**value2
    new_value2 = value2**value1

    new_tuple=(new_value1, new_value2)

    return new_tuple
raise_both(2,3)

# SCOPE in functions
# Not all objects area ccessible everywhere in a script
# Scope: part of the program where an object or name may be accesbile 
# 2 types of scopes: global,local
# Global: defined in the main body of a script
# Local: name defined inside a function (when the function is done, the name ceases to exist)
# Built-in-scope: names in th epre-defined built-ins module
# If Python cannot find a name in the local scope, it will look in the upper scope until it reaches global

# Alter the value of a GLOBAL name WITHIN a function call
new_val = 10
def square(value):
    """ Return the square of a value."""
    global new_val # searches new_val in global environment
    new_value= new_val**2
    return new_value
square(5)

# See Python's built-in scope
import builtins
dir(builtins)

# NESTED functions
def mod2plus5(x1, x2, x3):
    """Returns the remainder plus 5 of three values."""

    def inner(x):
        """Returns the remainder plus 5 of a value"""
        return x % 2 + 5
    
    return(inner(x1), inner(x2), inner(x3))
print(mod2plus5(1,2,3))

# A function to return a function
def raise_val(n):
    """Return the inner function."""

    def inner(x):
        """Raise x to the power of n."""
        raised = x ** n
        return raised

    return inner
square = raise_val(2)
cube = raise_val(3)
print(square(4), cube(3))

# Create and changed names in an enclosing scope (the 'upper' scope)
def outer():
    """Prints the value of n."""
    n = 1

    def inner():
        nonlocal n # n's value in the enclosing scope ('outer') will be modified but whatever is done in this scope 
        n = 2
        print(n)
    inner()
    print(n)
outer()

# DEFAULT ARGUMENTS
def power(number, pow=1): # pow takes a default value of 1
    """Raise number to the power of powe."""
    new_value = number ** powreturn new_value
power(9,2)

# FLEXIBLE ARGUMENTS
def add_all(*args): # *args turns all arguments passed to the funtion into a tupple
# Args do not need to have a keyword
    """Sum all values in *args together."""
    sum_all = 0
    for num in args:
        sum_all += num
    return sum_all
add_all(1,4,5)
add_all(1,4,5,123,-76)
# Flexible amount of KEYWORD ARGUMENTS
def print_all(**kwargs):
    """"Print out key-value pairs in **kwargs."""
    for key, value in kwargs.items():
        print( key + ': ' + value)
print_all(name='Harry', job='student')

# LAMBDA FUNCTIONS
# A quicker way to write functions
raise_to_power = lambda x, y: x ** y # arguments: instructions
raise_to_power(3,5)
# ANONYMOUS FUNCTIONS
# Pass functions to sequences using map
nums = [3,7,5,9]
square_all = map(lambda num: num ** 2, nums) # apply this lambda function to the list "nums"
print(square_all) # returns a map object
print(list(square_all)) # returns results in a list
# Apply a lambda function in a filter // filter the elements of a list according to a condition
list1 = ['asedfa', 'da', 'sadasda'] # eg return strings with more than 2 characters
result = filter(lambda string: len(string)>2, list1)
print(list(result))
# Reduce() and lambda functions
# Reduce loops through the elements of a list, and performs the function over the result of the previous iteration (algo así)
from functools import reduce
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']
result = reduce(lambda item1, item2: item1 + item2, stark)
print(result)

# ERROR HANDLING
# Catch exceptions with try-except clause
def sqrt(x):
    try:
        return x ** 0.5
    except TypeError: # here we catch only one kind of error
        print("x must be an int or float")
sqrt(1)
sqrt('1')
# Raise an error: when you wish your function not to work in specific circumnstances, regardless of whether Python would raise an error or not
sqrt(-1) # this runs ok with the previous function
def sqrt(x):
    if x<0:
        raise ValueError('x must be non-negative')
    try:
        return x ** 0.5
    except TypeError: # here we catch only one kind of error, but you can specify none, just write 'except'
        print("x must be an int or float")
sqrt(-1)


# %% PYTHON DATA SCIENCE TOOLBOX 2

## ITERATORS
# ITERABLE: an object that has an associated iter() method (an iterator)
# For example: lists, strings, dictionaries, range objects, and file connections are iterables, and have their associated iterators
# A for loop, for example, applies the function iter() "Under the hood" to an iterable object

# Iterate over a string
word = 'ask'
it = iter(word)
next(it)

# Star operator: unpack all elements of an iterator at once
word = 'ask'
it = iter(word)
print(*it)

# Iterate over the key-value pairs of a dictionary
dic = {'country':'ARG', 'city':'CABA'}
for key, value in dic.items():
    print(key, value)

# Iterate over file connections
file = open('file.txt')
it = iter(file)
print(next(it))

# ENUMERATE: takes an iterable as an argument and returns a tuple with pairs of the elements of the iterable and their index
list1 = ['vsd', 'asdq', 'asdf']
for index, value in enumerate(list1, start=0): # the default is to begin enumeration at 0
    print(index, value)

# ZIP: accepts an arbitrary number of iterables and returns an iterator of tuples
list1 = ['vsd', 'asdq', 'asdf']
list2 = ['ok', 'bye', 'hello']
print(list(zip(list1, list2)))
for z1, z2 in zip(list1, list2):
    print(z1, z2)
# Unpacking a zip object (retrieve original lists as tuples)
z1 = zip(list1, list2)
list1_ = zip(*z1)
print(list1_)

# Loading a file with iterators
# For example, to load heavy data into different chunks
# Here we calculate the sum of a column in different parts
import pandas as pd
total = 0
for chunk in pd.read_csv('file.csv', chunksize=1000):
    total =+ sum(chunk['col'])
print(total)

# LIST COMPREHENSIONS
# Format: [output expression for iterator in variable in iterable]
nums = [12, 8, 21, 3, 16]
new_nums = [num + 1 for num in nums]
print(new_nums)
# You can do this with any iterable. The components are: an iterable, an iterator variable (the members of the terable), and the output expression
# Create a list with a sequence of numbers
result = [num for num in range(11)] # from 0 to 11
print(result)
# Replace nested for-loops
[(num1, num2) for num1 in range(0,2) for num2 in range(6,8)]

# Nested list comprehensions: for example, create a matrix like:
matrix1 = [[0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 2, 3, 4]]
matrix2 = [[col for col in range(5)] for row in range(5)]
matrix1==matrix2

# Conditionals in comprehensions
[num ** 2 for num in range(10) if num % 2 == 0] # eg: square all even numbers from 0 to 9
[num ** 2 if num % 2 == 0 else 0 for num in range(10)] # eg: square all even numbers from 0 to 9, and output 0 for odds
# You can include conditions on the output expression, as above, and also on the iterable
# Format: [output expression conditional on output for iterator variable in iterable conditional on iterable]

# DICTIONARY COMPREHENSIONS
# eg. dictionary of positive integers as keys and their negative counterparts as values
{num: -num for num in range(10)}

# Create a dictionary froma zip object
list1 = ['vsd', 'asdq', 'asdf']
list2 = ['ok', 'bye', 'hello']
dict(zip(list1, list2))

# GENERATOR EXPRESSIONS / GENERATORS
# Definition: it does not store the list/dictionary in memmory. It returns a generator object over which we can iterate to produce the elements of the list/dictionary
result = (2 * num for num in range(10))
for num in result:
    print(num)
result = (2 * num for num in range(10))
print(list(result))
result = (2 * num for num in range(10))
print(next(result))
# This is an example of LAZY EVALUATION: you delay the evaluation of an expression until the value is needed

# GENERATOR FUNCTIONS: functions that produce generator objects
# Eg: return a generator of values from 0 to n
def num_sequence(n):
    """Generate values from 0 to n."""
    i= 0
    while i < n:
        yield i
        i += 1
result = num_sequence(10)
print(type(result))
print(next(result))

# Create a generator for a file (to stream data)
def read_large_file(file_object):
    """A generator function to read a large file lazily."""
    # Loop indefinitely until the end of the file
    while True:
        # Read a line from the file: data
        data = file_object.readline()
        # Break if this is the end of the file
        if not data:
            break
        # Yield the line of data
        yield data
# Open a connection to the file
with open('world_dev_ind.csv') as file:
    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)
    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))

# Stream data in chunks with Pandas
df_reader = pd.read_csv('ind_pop.csv', chunksize=10)
print(next(df_reader))
for chunk in df_reader:
    print(chunk)


# %% INTRODUCTION TO IMPORTING DATA IN PYTHON

# Two tipes of text files: table data (flat files like csvs) and plain text files

# READING A PLAIN TEXT FILE
filename = 'file.txt'
file = open(filename, mode='r') # 'r' is read only (write is 'w')
text = file.read() # assign the text to a variable applying the read method
file.close() # close connection to the file
print text()
# With a "with" statement you don't need to close the connection (best practice)
with open(filename, 'r') as file: # this is called a CONTEXT MANAGER
    print(file.read())
# Read one line at a tine
print(file.readline())

# READING FLAT FILES: .csv, .txt
# Definition: basic text files containing records (table data) without structured relationships. A record if a row of fields or attributes.
# It can have a header in the first row.
# It has delimiters
# It can be imported with numpy (if numerical) or pandas (to data frames)

# Importing flat files with Numpy
import numpy as np
data = np.loadtxt('file.txt', delimiter='\t', skiprow=1, usecols=[0, 2]) # tab delimiter
data = np.loadtxt('file.txt', dtype=str) # values will be imported as strings
# For mixed data types (structured arrays):
data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None) # names=True is used when there is a header
data = np.recfromcsv('titanic.csv', delimiter=',', names=True, dtype=None)

# Importing flat files with Pandas
import pandas as pd
pd.read_csv('file.csv')
pd.read_csv('file.csv', header=None) # default is first row as header
pd.read_csv('file.csv', nrows=5) # Choose the number of rows to read
pd.read_csv('file.csv', sep='\t') # choose delimiter
pd.read_csv('file.csv', comment='#') # Choose character for commented lines
pd.read_csv('file.csv', na_values=['NA', ' ']) # Choose character for missing values

# READING OTHER FILE TYPES: excel spreadsheets, Matlab, SAS, Stata, HDF5, pickled files

# IMPORTING PICKLED FILES
# Pickling is a way that Python has to store objects which do not have an obvious way of storing them (like lists or dictionaries).
# Instead of storing the object in a readable text format, it stores it as bytes.
import pickle
with open('file.pkl', 'rb') as file: # 'rb' is read binary
    data = pickle.load(file)
print(data)

# IMPORTING EXCEL SPREADSHEETS
import pandas as pd
file ='file.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names) # print sheet names
df1 = data.parse('sheet1', skiprows=[0], names=['col1', 'col2', 'col3']) # load a specific sheet by sheet name
df1 = data.parse(0) # load a specific sheet by sheet index
# More direct with pd.read_excel
df = pd.read_excel(file, sheet_name=None) # if sheet name is null, it loads all sheets into a dictionary

# Get current working directory
import os
os.getwd()
# Print the content of the current working directory
os.listdir(os.getwd())

# IMPORTING FILES FROM SAS
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('file.sas7bdat') as file:
    df_sas = file.to_data_frame()

# IMPORTING FILES FROM STATA (.dta files)
import pandas as pd
data = pd.read_stata('file.dta')

# IMPORTING HDF5 FILES: hierarchical data format version 5 (for storing large quantities of numerical data of hundreds of GB or TBs or HBs)
import h5py
data = h5py.File('file.hdf5', 'r')
# The structure of HDF5 files:
for key in data.keys():
    print(key)
for key in data['col'].keys():
    print(key)
print(np.array(data['col']['subcol1']), np.array(data['col']['subcol2']))

# IMPORTING MATLAB FILES (.mat files)
import scipy.io
mat = scipy.io.loadmat('file.mat') 

# INTRODUCTION TO RELATIONAL DATABASES
# The relational model is defined by Codd's 12 rules
# PostgreSL, MySQL, SQLite etc.

# CREATING AN SQL DATABASE ENGINE IN PYTHON
# Here we will use an SQLite database and the package SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///databasename.sqlite')
# The engine will communicate to the database
# Get the names of the tables in the database
engine.table_names()
# Querying the database
# Basic SQL query
SELECT * FROM Table_Name # returns all columns of all rows of the table. The * means all columns
# But to use this in Python, first you need to import required packages and functions, create the database engine, connect to it, query the database, save the results to a df, and close the connection
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///databasename.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Table_Name") # here you execute the query
df = pd.DataFrame(rs.fetchall()) # Fetch all rows
print(df.head()) # Realize that column names are wrong
df.columns = rs.keys() 
con.close()
# Use the CONTEXT MANAGER, you don't need to close the connection at the end
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///databasename.sqlite')
with engine.connect() as con:
    rs = con.execute("SELECT Col1, Col2 FROM Table_Name WHERE ColumnName = 'Value'") # in this case we selected specific columns from the table
    df = pd.DataFrame(rs.fetchmany(5)) # And we imported just 5 rows
    df.columns = rs.keys()
# Another query as example
rs = con.execute("SELECT * FROM Customer ORDER BY SupportRepId")
# USING THE PANDAS FUNCTION read_sql_query               
import pandas as pd
engine = create_engine('sqlite:///databasename.sqlite')
df = pd.read_sql_query("SELECT * FROM TableName", engine)
# ADVANCED QUERYING: QUERYING MULTIPLE TABLES
# JOIN
pd.read_sql_query("SELECT * FROM Table1_Name INNER JOIN Table2_Name on Table1_Name.PKeyColName = Table2_Name.PKeyColName", engine)

# %% INTERMEDIATE IMPORTING DATA IN PYTHON
# The focus is on importing data from the Web (scraping)
# Load this data directly into pandas data frames
# Making HTTP GET requests
# Scrape web data such as HTML
# Parse HTML with BeautifulSoup library
# Use the urllib and request packages

# URLLIB PACKAGE
# urlopen() accepts URLs instead of file names
from urllib.request import urlretrieve
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
urlretrieve(url, 'winequality-white.csv')
# You can also do it with pandas's read_csv
pd.read_csv(url, sep=',')

# HTTP REQUESTS 
# ULR stands for "Universal Resource Locator". Most are website locations/web addresses
# They have two parts: a protocol identifier ("http or https") and a resource name ("google.com")
# HHTP stands for "HyperText Transfer Protocol", it is an application protocol for distributed, collaborative hypermedia information systems (like Wikipedia). It is the foundation of data communication for the WWW.
# HHTPS is a more secure form of HHTP.
# Each time you go to a website you are actually sending an HTTP request to a server (like a GET request). The function urlretrieve() makes a get request and saves the data locally.
# HTML stands for "HyperText Markup Language" and is the standard markup language for the web.
# With URLLIB package
from urllib.request import urlopen, Request # import necessary functions
url = "https://wikipedia.org/" # specify the url
request = Request(url) # package the get request using the function Request
response = urlopen(request) # send the request and catch the response. This returns an HTTP response object
html = response.read() # Apply the associated "read" method to the HTTP response object, which returns the HTML as a string
print(html)
response.close() # close the response
# With REQUEST package (an API for making requests)
import requests
url = "https://wikipedia.org/"
r = requests.get(url) # with the get function you package, send and catches the response
text = r.text # return the html as a string
print(text)

# BEAUTIFUL SOUP: web scraping, parsing structured data from HTML
# After you have made a get request, the data is still in an unreadeable format. BeautifulSoup can help with that.
from bs4 import BeautifulSoup
import requests
url = "https://wikipedia.org/"
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc) # Once you have retrieved the html, run this fnction to parse it
print(soup.prettify()) # The prettify method indents the text as it would be in html code
print(soup.title) # Extract html title
print(soup.get_text()) # Extract html text
# Extract urls of all hyperlinks in the html:
for link in soup.find_all('a'): # hyperlinks are defined by the HTML tag <a>
    print(link.get('href'))

# INTRODUCTION TO APIs and JSONs
# API: application programming interface. It is a set of protocols and rutines for building and interacting with software applications.
# A bunch of code that allows two software programmes to communicate with each other.

# JSON: JAvaScript object notation. A file format which is a standard form for transferring data through an API.
# They are human readable. It conssits of name-value pairs separated by commas.

# Loading JSON files
import json
with open('file.json', 'r') as json_file: # open a connection to the file
    json_data = json.load(json_file) # use json.load() to load it
print(type(json_data)) # it imports it as a dictionary
for key, value in json_data.items(): # iterate across key-value pairs and print content
    print(key + ':' + value)

# Connecting to an API using requests package
import requests
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
# First put a '?' sign, then put the API key, and "t=the+social+network" is a query string, where 't' means title
r = requests.get(url)
json_data = r.json() # convert the response object to a dictionary with the json() decoder
for key, value in json_data.items():
    print(key + ':', value)

# Twitter API
# Twitter has more than one API
# REST API: allows the user to read and write twitter data
# Public streams: one of the streaming APIs which gives you access to streams of the public data.
# It cointains different options. We will beusing the GET statuses/sample API, which returns a small random sample of public tweets.
# To access more data, one could use the firehouse API, which is paid.
# ACcess stream data from the Twitter API with Tweepy package
import tweepy, JSON
acces_token = '...'
access_toke_secret = '...'
consumer_key = '...'
consumer_secret = '...'
# Create a streaming object
stream = tweepy.Stream(consumer_key, consumer_secret, access_token, access_token_secret)
# Filter tweets with certain keywords
stream.filter(track=['apples', 'oranges'])


# %% CLEANING DATA IN PYTHON

# Pandas stores strings as the data type "object"

# Get data frame info
df.info()

# Get summary statistics of a column
df['col'].describe()

# Clean a string variable to make it integer
# Remove a character from a string variable in pandas
df['col'] = df['col'].str.strip('$')
# Convert a string column to integer
df['col'] = df['col'].astype('int') # change the data type of a variable

# Clean an integer variable to make it categorical
df['col'] = df['col'].astype('category')

# Get the data type of a column
df['col'].dtype

# DATA RANGE CONSTRAINTS

# Check if there are dates posterior to today's date
import datetime as dt
# Convert string column to date
df['date'] = pd.to_datetime(df['date']).dt.date
today_date = dt.date.today()
df[df['datecol'] > today_date]
# Drop them
df.drop(df[df['date']>today_date, 'date'].index, inplace=True)

# Drop values that are incorrect
# Using filtering
df = df[df['rating'] <= 5]
# Using .drop() method
df.drop(df[df['rating'] > 5].index, inplace=True)

# Replace row values according to a condition
df.loc[df['rating'] > 5, 'rating'] = 5


# UNIQUENESS CONSTRAITNS / DUPLICATE VALUES
# Get duplicates across all columns
df[df.duplicated()]
# Get duplicated across specific columns
duplicates = df.duplicated(subset=['col1', 'col2'], keep='first') # subset arg let's you choose columns. keep arg lets you keep first (default), last or all (False) duplicate values.
df[duplicates].sort_values(by = ['col1', 'col2'])
# Drop duplicates
df.drop_duplicates(inplace = True) # across all columns
# Replace duplicates with the mean of the duplicated values
col_names = ['col1', 'col2']
summaries = {'col1': 'mean', 'col2':'max'}
df = df.groupby(by = col_names).agg(summaires).reset_index()
df[duplicates].sort_values(by = ['col1', 'col2'])

# CATEGORIES AND MEMBERSHIP CONSTRAINTS
# Possible solutions, dropping data, remapping categories, inferring categories
# Find inconsistent categories (comparing the values of a column with a list with all possible valid values (or a df column))
inconsistent_categories = set(df['cat1']).difference(categories_df['cat1'])
# Find the rows with these inconsistent categories
inconsistent_rows = df['cat1'].isin(inconsistent_categories) # boolean mask
df[inconsistent_rows]
# Drop rows with inconsistent categories
df = df[~inconsistent_rows]


# VALUE INCONSISTENCY
# Eg string values that differ due to capital letters
df['cat1'].value_counts()
df.groupby('cat1').count()
df['cat1'] = df['cat1'].str.lower()
df['cat1'].value_counts()

# Remove leading and trailing blank spaces
df = df['col'].str.strip()

# Create categories out of data. Two ways
# a) Using the qcut() method from Pandas
import pandas as pd
group_names = ['0-100', '100-200', '200-300']
df['group_col'] = pd.qcut(df['var'], q=3, labels=group_names) # equally spaced groups
df[['group_col', 'var']]
# b) Using cut() function from Pandas to create category ranges and names
ranges = [0, 100, 200, np.inf]
group_names = ['0-100', '100-200', '200-Inf']
df['group_col'] = pd.cut(df['var'], bins=ranges, labels=group_names) # group ranges can be set
df[['group_col', 'var']]

# Mapping categories to fewer ones
df['col'].unique()
mapping = {'AB': 'A', 'AC':'A', 'BB':'B'}
df['col'] = df['col'].replace(mapping) # .replace(): pandas method to replace values of a column
df['col'].unique()

# CLEANING TEXT DATA
# Replace strings
df['col'] = df['col'].str.replace('-', ',')
df['col'] = df['col'].str.replace('_', ',')
# Check
assert df['col'].str.contains("-|_").any() == False # check whether a string is present in a column
# Correct length violation
digits = df['phone_num'].str.len()
df.loc[digits z 10, 'phone_num'] = np.nan
# Check
assert df['phone_num'].min() >= 10

# Using REGULAR EXPRESSIONS to replace anything that is not a digit/number with a blank space
df['col'] = df['col'].str.replace(r'\D+', '')

# DEALING WITH DATES IN MULTIPLE FORMATS AND WITH ERRORS
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce') # returns NaT for rows where the conversion failed

# Convert the format of a date time column
df['date'] = df['date'].dt.strftime("%d-%m-%Y") # change the format of a date
df['date'].dt.strftime("%Y") # extract the year / month / day out of a date

# Calculate the years from a date until today
dt.date.today().year - df['date'].dt.year

# COMPLETENESS AND MISSING DATA
# Return boolean masks for missing values in all columns
df.isna()
# Summary of missing values by column / quantity of missing values by volumn
df.isna().sum()
# Check out "missingno" package. Missing value summary plot
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df)
plt.show()
# You can sort the rows according to a column if you think that missingness is not random but varies with that column
msno.matrix(df.sort_values(by='col'))
plt.show()
# Select all rows with missing values in a data frame
missing = df[df['col'].isna()]
missing.describe
complete = df[~df['col'].isna()]
complete.describe
# Delete rows with missing values in one column
df = df.dropna(subset = ['col']) # or
df.dropna(subset=['col'], inplace=True)
# Replace missing values with a statistical measure
df = df.fillna({'col_with_missing': df['col_with_missing'].mean()})
# Replace missing values with previous or posterior non missing row
df = df['col_with_missing'].fillna('backfill') # use next valid observation to fill gap
df = df['col_with_missing'].fillna('ffill') # propagate last valid observation forward to next valid

# COMPARING STRINGS
# MINIMUM EDIT DISTANCE: the least possible amount of steps needed to transition from one string to another.
# The "valid" operations are: inserting, deleting, substitution and transpositioning consecutive characters
# It is a measure of how close two strings are.
# Simple string comparison (using Levenshtein distance) with "thefuzz"/FuzzyWuzzy package 
from thefuzz import fuzz
fuzz.WRatio('Reeding', 'Reading') # Outputs a score from 0 to 100, where 100 is an exact match
fuzz.WRatio('Houston Rockets', 'Rockets')
# Compare a string with an array of possible matches
from thefuzz import process
import pandas as pd
string = 'Houseton Rockets vs Los Angeles Lakers'
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets', 'Houston vs Los Angeles', 'Heat vs Bulls'])
process.extract(string, choices, limit=2) # It returns a list of tuples with the matching string being returned, the similarity score, and its index in the array
# String similarity if very useful when there are so many errors in a category that manual replacement is unfeasible
# Eg wrong state names
categories = ['correct_state_name1', 'correct_state_name2']
# For each correct category
for state in categories:
    # Find potential matches in states with typoes
    matches = process.extract(state, df['state'], limit = df.shape[0]) # set the limit argument of the extract function to the length of the data frame to get all possible matches for each category (there can be no more matches than rows in the df)
    # For each potential match
    for potential_match in matches:
        # If it has a high similarity score (more than 80)
        if potential_match[1] >= 80
            # Replace typo with correct category
            df.loc[df['state'] == potential_match[0], 'state'] = state

# GENERATING PAIRS
# Joining corresponding rows between data frames when IDs are not identical, so a regular merge will not work.
# Record linkage is the act of linking data from different sources regarding the same entity.
# The process involves generating pairs, comparing them, scoring pairs, and choosing the pairs with the highest scores to link the data.
# We can do this with the 'recordLinkage' package
# Generating pairs by 'blocking', which creates pairs based on a matching colun, reducing the number of possible paris
import recordLinkage
indexer = recordlinkage.Index() # create an indexing object used to generate pairs from the df
indexer.block('col1') # Generate blocked pairs based on 'col1'
pairs = indexer.index(dfA, dfB)
type(pairs) # Pandas multi-index objects. 
print(pairs) # An array with possible pair of indices.
# Pairs are already generated, time to find potential matches
compare_cl = recordlinkage.Compare() # create a comparison object. assigns different comparison procedures for pairs
# Find exact matches of pairs of certain columns (here date_of_birth and state)
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth') # the label arguments let's you set the name of the column in the resulting df
compare_cl.exact('state', 'state', label='state')
# For columns with fuzzy values, find similar matches for pairs of certain columns (here 'surname' and 'address_1') using string similarity
compare_cl.string('surname', 'surname', threshold=0.85, label='surname') 
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')
# Find matches
potential_matches = compare_cl.compute(pairs, dfA, dfB)
# !!! The order of the data frames has to be always the same
print(potential_matches) # The result gives, for each row index in dfA, a list with all row indices from dfB that could be a pair.
# The columns are the columns of the dfs being compared. 1 represents a match, 0 not a match.
# To find potential matches, you filter for rows where the sum of row values is higher than a certain threshold
matches = potential_matches[potential_matches.sum(axis=1) >= 2]
# LINKING DATA FRAMES ACCORDING TO GENERATED PAIRS
# Once you have potential matches, the next is to extract one of the index columns and subsetting its associate df to filter for duplicates
# For example, chose the second index column. Extract them and subset dfB on them to remove duplicates with dfA before appending them together
matches.index
# Get indices from dfB only
duplicate_rows = matches.index.get_level_values(1) # 1 is the order of the column with the indices
# Finding duplicates in dfB
dfB_duplicates = dfB[dfB.index.isin(duplicate_rows)]
# Finding new rows in dfB
dfB_new = dfB[~dfB.index.isin(duplicate_rows)]
# Link the data frames
full_df = dfA.append(dfB_new)


# %% WORKING WITH DATES AND TIMES IN PYTHON

# Special Python class: "date"

# CH1: WORKING WITH DATES AND CALENDARS

# Create a date object by hand
from datetime import date
dates = [date(2016, 10, 7), date(2017, 6, 21)] # Y, M, D
# Access components of a date using dates attributes
dates[0].year
dates[0].month
dates[0].days
dates[0].weekday() # Monday is 0, Sunday is 6

# MATH WITH DATES
# Substract two dates
dates = [date(2016, 10, 7), date(2017, 6, 21)]
date(2017, 6, 21) - date(2016, 10, 7) # returns an object of type timedelta (the ellapsed time between events)
timediff = date(2017, 6, 21) - date(2016, 10, 7)
timediff.day # days ellapsed
# Minimum date
min(dates)
# Create a timedelta, add it to a date
from datetime import timedelta
td =timedelta(days=29)
print(date(2017, 6, 21) + td)

# Turning dates into strings (eg to write them to a csv fileimport date from datetime
d = date(2017, 11, 5)
print(d) # ISO format: ////-MM-DD
# Get the ISO representation fo a date as a string
d.isoformat()
# For every other format, you can format your data with
d.strftime("%Y")
d.strftime("Year is %Y") # very flexible
d.strftime("%Y/%m/%d")
d.strftime('%B (%Y)')
d.strftime("%Y-%j") # %j gives day as number of day in the year


# CH2: COMBINING DATES AND TIMES

# Create a datetime object
from datetime import datetime
dt = datetime(2017, 10, 1, 15, 23, 25, 500000) # year, month, day, hour, minutes, seconds, microseconds. All have to be whole numbers
print(datetime(2017, 10, 1, 15, 23, 25, 500000).hour)
print(datetime(2017, 10, 1, 15, 23, 25, 500000).minute)
print(datetime(2017, 10, 1, 15, 23, 25, 500000).second)
print(datetime(2017, 10, 1, 15, 23, 25, 500000).microsecond)
# Create a datetime object from an existing one / replace values of a datetime object
dt_hr = dt.replace(minute=0, second=0, microsecond=0)

# Printing datetimes
dt = datetime(2017, 12, 30, 15, 19, 12)
print(dt.strftime("%Y-%m-%d %H:%M:%S")) # year, monht, day hour, minutes, seconds
print(dt.isoformat())

# Parse dates from strings
from datetime import datetime
dt = datetime.strptime("12/30/2017 15:19:13", "%m/%d/%Y %H:%M:%S") # 'strptime' is short for "string parse time". The first arg is the string we want to parse, the second is the format
dt = datetime.strptime("12/30/2017 15:19:13", "%Y-%m-%dT%H:%M:%S") # re create the ISO format

# Parsing dates from Unix timestamps: (the number of second since January 1st 1970)
ts = 1514665153.0
print(datetime.fromtimestamp(ts))

# WORKING WITH DURATIONS (timedelta)
start = datetime(2017, 10, 8, 23, 46, 47)
end = datetime(2017, 10, 9, 0, 10, 57)
duration = end - start
duration.total_seconds() # get total seconds ellapsed

# Create a timedelta object by hand
delta1 = timedelta(seconds=1) # add a timedelta
print(start + delta1)
delta2 = timedelta(days=1, seconds=1) # substract a timedeltan
print(start - delta2)
delta3 = timedelta(weeks=-1) # negative durations are allowed
print(start + delta3)

# CH3: TIME ZONES AND DAYLIGHT SAVING

# UTC OFFSETS: compare times around the world
# UTC: coordinated universal time
# Create a timezone object
from datetime import datetime, timedelta, timezone
# US Eastern Standard time zone
ET = timezone(timedelta(hours=-5)) # This is UTC - 5
UTC = timezone.utc # UTC
# Create a datetime object specifying the time zone
dt = datetime(2017, 12, 30, 15, 9, 3, tzinfo = ET)
print(dt) # the UTC offset is the "-05:00" at the end of the datetime
# Convert datetimes to another time zone
IST = timezone(timedelta(hours=5, minutes=30)) # India Standard time zone, UTC + 5.30
dt.astimezone(IST)
# You can also do it doing the replace method
dt.replace(tzinfo=IST)
# The difference is that the replace method has to be used just for "one-off" changes, eg to a particular value, while changing the timezone with astimezone supplies it as "context information" and is able to pick the correct offset applicable to the datetime

# TIME ZONE DATABASE: tz database, updated 4 times a year
# Get a time zone from the time zone database using "dateutil" package
from datetime import datetime
from dateutil import tz
et = tz.gettz("America/New_York") # Format: "Continent/City"
tz.gettz('UTC')
# It takes into account the daylight saving time
print(datetime(2017, 12, 30, 15, 9, 3, tzinfo=et))
print(datetime(2017, 10, 1, 15, 23, 25, tzinfo=et)) # see how the UTC offset changed because of DST

# Starting daylight saving time
# Eg: clocks move forward in the spring, like on March 12 2017 in Washington DC
# The UTC object will now have to change to reflect this
# Lets start by creating the offset by hand, then we will use dateutil
from datetime import datetime, timedelta, timezone
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59) # 1:59
spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0)
(spring_ahead_3am - spring_ahead_159am).total_seconds() # an hour and one second apart
EST = timezone(timedelta(hours=-5))
EDT = timezone(timedelta(hours=-4))
spring_ahead_159am = spring_ahead_159am.replace(tzinfo = EST)
spring_ahead_159am.isoformat()
spring_ahead_3am = spring_ahead_3am.replace(tzinfo = EDT)
spring_ahead_3am.isoformat()
(spring_ahead_3am - spring_ahead_159am).seconds # only one second apart
# The problem with this approach is that you have to know when the cut off happened. dateutil solves this for us
from dateutil import iz
easter = tz.gettz("America/New_York")
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59, tzinfo=eastern)
spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0, tzinfo=eastern) # dateutils figures out that the tz should be EDT and not EST

# Ending daylight saving time (when clocks fall back to standard time in the fall)
from datetime import datetime, timedelta, timezone
eastern = tz.gettz("US/Eastern")
first_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
# Check if the datetime is ambiguous (that could occur at 2 utc moments in the time zone)
tz.datetime_ambiguous(first_1am)
second_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo=eastern)
second_1am = tz.enfold(second_1am) # this says that the datetime belongs to the "second" timezone, the one after DST
(first_1am - second_1am).total_seconds() # it doesn't change basic operations
# The BEST PRACTICE is to convert them to UTC which is an unambiguos timezone
first_1am = first_1am.astimezone(tz.UTC)
second_1am = second_1am.astimezone(tz.UTC)
(first_1am - second_1am).total_seconds()


# CH4: DATES AND TIMES IN PANDAS

import pandas as pd
# Load a csv and parse dates as datetimes (actually a Pandas Timestamp)
df = pd.read_csv('file.csv', parse_dates = ['start_date', 'end_date'])
# Or change a columnd to a datetime format manually
df['start_date'] = pd.to_datetime(df['start_date'], format = "%Y-%m-%d %H-%M-%S")
# You can substract columns and get timedeltas
df['duration'] = df['end_date'] - df['start_date']
# Convert a column to seconds 
df['duration'] = df['duration'].dt.total_seconds()

# SUMMARIZING DATETIME DATA IN PANDAS
df['duration'].mean()
df['duration'].sum()
df['duration'].sum() / timedelta(days=91)
df['duration'].total_seconds()
df['duration'].total_seconds().min()
df['duration'].dt.year
df['duration'].dt.month
df['duration'].dt.day
df['duration'].dt.day_name() # returns the name of the day of the week (available in other langauges)
df['duration'].shift(1) # this makes the value in the n row move to the n+1 row, and row zero is left with Na
# Average of a column by period using resample / group by the period attribute of a datetime column
# Resampling: when you change the frequency of time-series observations
df.resample('M', on = 'start_date')['col'].mean() # by month, group on a datetime column
# If the df index was the date column, the 'on' argument would be unnecessary
df.resample('M', on = 'start_date')['col'].mean() # by days
df.resample('M', on = 'start_date')['col'].mean().plot()
# It is equivalent to a groupby operation on df.index.month, excep that the resulting index for each row is the last day of each month

# WARNING: take into account TIME ZONES and DAYLIGHT SAVING TIME
# Set the time zone of a datetime column
df['duration'].dt.tz_localize("America/New_York")
# However, there may be datetimes that occur just during a shift, so it is not unambiguous
df['duration'].dt.tz_localize("America/New_York", ambiguous='NaT') # sets ambiguous rows as datetime missing values ('not a time')
df['duration'].dt.tz_convert('Europe/London')
df['duration'] = df['end_date'] - df['start_date']


# %% WRITING FUNCTIONS IN PYTHON

### CH1: BEST PRACTICES


# DOCSTRINGS
# A string written as the first line/s of a function, enclosed in triple quotes, which usually explains:
# 1) What the function does
# 2) What the arguments are
# 3) What the return value/s should be
# 4) Info about any errors raised
# 5) Something else about the function
# THere are serveral standards for how to write docstrings: Google Style, Numpydoc, reStructuredText, EpyText

# Google Style format: the docstirng starts with an imperative description of what the function does, in imperative terms.
# Then comes the "Args" section where each argument name is listed followed by its expected type in parenthesis, and what its role is in the fucntion, as well as default arguments
# Return section: lists the expected type or types of what gets returned, as well as some comment about it
# Raises section: specify if the function intentionally raises any error
# Notes section: additional notes or examples of usage in free-form text
def function(arg1, arg2=43):
    """Description of what the function does.
    
    Args:
        arg1 (str): Description of arg1 that can break onto the next line if needed.
        arg2 (int, optional): Write optional when an argument has a default value.end=
    
    Returns:
        bool: Optional description of the return values
        Extra lines are not indented.

    Raises:
        ValueError: Include any error types that the funtion intentionally raises.

    Notes:
        See ... for more info
    """

# Numpydoc format
# The most common format in the scientific community, but takes more vertical space.
def function(arg1, arg2=43):
    """
    Description of what the function does.

    Parameters
    ----------
    arg1 : expected type of arg1
        Description of arg1
    arg2: int, optional
        Write optional when an argument has a default value.
        Default=43

    Returns
    -------
    The type of the return value
        Can include a description of the return value.
        Replace "Returns" with "Yields" if this function is a generator.
    """

# How to retrieve the docstring of a function: with the __doc__ attribute
print(function1.__doc__) # raw docstring, including tabs of spaces
# For a cleaner version of the docs
import inspect
print(inspect.getdoc(function1)) # a clean

# TWO PRINCIPLES TO FOLLOW:
# 1) DRY ('don't repeat yourself'): when you are using the same chunk of code over and over for different data sets, it is probably better to write a function that does that
# 2) Do one thing: each function should have a single responsibility

# REFACTORING: the process of improving code by changing it a little bit at a time

# PASS BY ASSIGNMENT
# The way Python passes argument to a function is particular
# Mutatable data types can be modified by passing them as arguments in a function (without returning anything) (sort of), while inmutable data types cannot.
# Immutable data types: int, float, bool, string, bytes, tuple, frozenset, None
# Mutable data types: list, dict, set, bytearray, objects, function, almost everything else
# The only way to check whether something is mutable is to see if there is a function or method that will change the object without assigning it to a new variable
# When writing a function with a default value, try not to put a mutable object as the defualt value, because it may happen that it gets changed when running the function, and it is not reasonable to have a default value that changes


### Ch2: CONTEXT MANAGERS

# A context manager is a type of function that sets up a context for your code to run in, runs your code, and then removes the context.
# One example is the open function, like when you use
with open('file.txt') as my_file:
    text = my_file.read()
    length = len(text)
print('The file is {} characters long'.format(length))
# Here open() does 3 things: it sets up a context by opening a file that you can read or write on, then it gives control back to your code so that you can perform operators on the file object. Finally, when the code inside the indented section is done, the file is closed. By the time thte print statement is run, the file is closed.
# Using a context manager. Always starts with 'with', followed by a function that is a context manager
with <context-manager>(<args>):
    # Run code here
    # This code is running "inside the context"
# This code runs after the context is removed
# Statements that have an indented block after them are called 'compound statements'
# Some context managers want to return a variable that you can use inside the context
# By adding 'as' and a variable name at the end of the with statement you can assign the return value as a variable name

# Writing context managers / creating functions that are context managers
# There are two ways to do this:
# 1) Class-based: using a class that has __enter__ and __exit__ methods
# 2) Function-based: decorating a certain kind of function (here we see only this one)
# 5 steps:
# -1 define a function
# -2 (optional) add any set up code your context needs
# -3 use the "yield" keyword to signal Python that this is a special kind of function
# -4 (option) Add any teardown code that you need to clean up the context
# -5 Add the '@contextlib.contextmanager' decorator
import contextlib
@contextlib.contextmanager
def my_context():
    print('hello')
    yield 42 # used when you are going to return a value but you expect to finish the rest of the function at some point in the future
    # The yielded value can be assigned to a variable in the 'with' statement by adding 'as var_name'
    # In fact, a context manager function is technically a generator that yields a single value
    print('goodbye') 
with my_context() as foo:
    print('foo is {}'.format(foo))
# Another example
@contextlib.contextmanager
def database(url):
    # set up database connection
    db = postgres.connect(url)

    yield db
    # tear down database connection
    db.disconnect()
url = 'http://datacamp.com/data'
with database(url) as my_db:
    course_list = my_db.execute('SELECT * FROM courses')
# The "set up - tear down" behaviour allows a context manager to hide things like connecting and disconnecting to a db so that a programmer working using the context manager can just perform operations on the db without worrying about underlying details

# Some context managers do not yield a specific value
@contextlib.contextmanager
def in_dir(path):
    # save current working directoy
    old_dir = os.getwd()

    # switch to new working directory
    os.chdir(path)

    yield

    # change back to previous working directory
    os.chdir(old_dir)

with in_dir('/data/project_1/'):
    project_files = os.listdir()

# Timer / function that times code
import time
@contextlib.contextmanager
def timer():
  """Time the execution of a context block.

  Yields:
    None
  """
  start = time.time()
  # Send control back to the context block
  yield
  end = time.time()
  print('Elapsed: {:.2f}s'.format(end - start))

with timer():
  print('This should take approximately 0.25 seconds')
  time.sleep(0.25)

# Read-only version of the open() context manager
@contextlib.contextmanager
def open_read_only(filename):
  """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
  read_only_file = open(filename, mode='r')
  # Yield read_only_file so it can be assigned to my_file
  yield read_only_file
  # Close read_only_file
  read_only_file.close()

with open_read_only('my_file.txt') as my_file:
  print(my_file.read())

# NESTED CONTEXTS
# For example, if you write a function that copies the content of one file to another file, one option would be to
# open the source file, store the content in memory, and then write the contents to the other file
# However, if the file is too large, this is not very efficient because it requires storing the file in memmory.
# Ideally we could open both files at once and copy one line at a time, like this
def copy(src, dst):
    '''
    Copy the contents of one file to another.

    Args:
        src (str): File name of the file to be copied.end=
        dst (str): Where to write the new file.
    '''

    # Open both files
    with open(src) as f_src:
        with open(dst, 'w') as f_dst:
            # Read and write each line, one at a time
            for line in f_src:
                f_dst.write(line)

# HANDLING ERRORS
# If the user, when using a context manager, runs a code that returns an error, if it is not taken into acount, the teardown code will not be run (for example, the file will not be closed).
# To take this into account, one can take advantage of the try-finally combination
# For example, a made-up function that opens the connection to a printer, prints a text, and then closes the connection, since only one user can be connected at a time.
def get_printer(ip):
    p = connect_to_printer(ip)

    try:
        yield
    finally: # 'finally' precedes a code that will be run regardless of whether the code in 'try' run or returned an error
        p.disconnect()
        print('disconnected from printer')

doc = {'text': 'This is my text.'}
with get_printer('10.0.34.111') as printer:
    printer.print_page(doc['txtt']) # here we mistook 'text' for 'txtt', so the function should return an error, but still close the connection

# COMMON PATTERS THAT REQUIRE A CONTEXT MANAGER
# IF you notice that your code is following any of the following patterns, you might consider using a context manager:
# - Open-Close
# - Lock-release
# - Change-reset
# - Enter-exit
# - Start-stop
# - Setup-teardown
# - Connect-disconnect


### Ch3: DECORATORS

# FUNCTIONS AS OBJECTS: functions are not fundamentally different from any other Python object
# Therefore, you could do things to it like you would do to any other object,
# like assigning it to another variable
def my_func():
    print('Hello')
x = my_func # do not include the parenthesis to refference the function itself, otherwise you are calling it and it evaluates to the value that the function returns
print(type(x))
x()
# or add them to a list
list_of_funcs = [my_func, open, print]
list_of_funcs[2]('Alo')
# or to a dictionary
dict_of_funcs = {'f1': my_func, 'f2': open, 'f3':print}
dict_of_funcs[2]('Alo')
# Pass a function as an argument of another function
def has_docstring(func):
    """Check to see if the function 'fun' has a docstring.
    
    Args:
        func (callable): A function.
        
    Returns:
        bool
    """
    return func.__doc__ is not None
has_docstring(print)

# NESTED FUNCTIONS / INNER FUNCTIONS / HELPER FUNCTIONS / CHILD FUNCTIONS: functions defined inside another function
# Sometimes they make your code easier to read
def foo(x, y):
    if x >4 and z < 10 and y > 4 and y < 10
    print(x * y)
# Can be improved into
def foo(x, y):
    def in_range(v):
        return v > 4 and v < 10
    if in_range(x) and in_range(y):
        print(x * y)

# A function that returns a function
def get_function():
    def print_me(s):
        print(s)
    return print_me
new_f = get_function()
new_f('Print this')

# SCOPE: determines which variables can be accessed at different points in your code
# Let's use an example
x = 7
y = 200
print(x)
# What happens if you redefine X inside a function
def foo():
    x = 42
    print(x)
    print(y)
# Here, the function prints the x that was defined inside it. Also, as there is no Y defined inside it looks outisde the function for a definition.
# Note that setting the value of x = 42 inside the function does not changed the value of X defined outside the function.
print(x)
# The formal rules that set this behaviour are:
# 1st: look in the LOCAL scope
# 2nd: if it cannot find it, search in the NONLOCAL scope (the PARENT function)
# 3rd: if it cannot find it, search in the GLOBAL scope
# 4th: if it cannot find it, search in the BUILT IN scope (things like the 'print' function)

# GLOBAL KEYWORD: change the value of an object from the global scope, inside a function
x = 7
def foo():
    global x 
    x = 42
    print(x)
foo()
# use 'nonlocal x' for nonlocal objects
def foo():
    x=10
    def bar():
        nonlocal x
        x=200
        print(x)
    bar()
    print(x)
foo()

# CLOSURES: a tuple of variables that are no longer in scope but that a function needs in order to run
# It is a way to attach nonlocal variables to a returned function so that the function can operate even when it is called outisde of its parent scope
# Attaching nonlocal
def foo():
    a=5
    def bar():
        print(a)
    return bar
func = foo()
func()
# When foo returned the new 'bar' function, Python attached any nonlocal variable that 'bar' was going to need to the function object.
# Those variables get stored in  atuple i the __closure__ attribute of the function
print(type(func.__closure__))
len(func.__closure__)
# You can access the cell contents of the closure
func.__closure__[0].cell_contents

# Closures and deletion
x = 25
def foo(value):
    def bar():
        print(value)
    return bar
my_func = foo(x)
del(x)
my_func()
# Even though we deleted X from the global scope, my_func() still knows its value because foo's value argument was added to the closure attached to my_func function.
#So even X doesn't exist anymore, the value persists in the closure.
# Closures and overwriting
x = 25
def foo(value):
    def bar():
        print(value)
    return bar
x = foo(x)
x()
# Still returns 25 even though x value was overwritten in the global scope

#All this matters because DECORATORS need to use all of the following in order to work:
# Functions as objects, nested funcitons, nonlocal scope, closures

# DECORATORS
# A decorator is a wrapper that you can place around a function that changes its behaviour
# You can modify the inputs, the outputs, and/or the behaviour of the function
# It is a function that takes a function as an argument and returns a modified version of it
# For example, let¡s build a 'double args' decorator that multiplies every argument by 2 before passing it to the decorated function
def multiply(a, b):
    return a * b
def double_agrs(func):
    # Define  anew function that we can modify
    def wrapper(a, b):
        # Call the passed in function, but double each argument
        return func(2*a, 2*b)
    # Return the new function
    return wrapper
new_multiply = double_args(multiply)
new_multiply(1, 5)
# Try this
multiply = double_args(multiply)
multiply(1, 5) # returns 20, because the original multiply function is stored in the new function's closure
multiply.__closure__[0].cell_contents
# Now using the syntax for decorators
@double_args # this is equivalent to 'multiply = double_args(multiply)'
def multiply(a, b):
    return a*b
multiply(1, 5)


# Ch5: REAL-WORLD EXAMPLES OF DECORATORS

# TIMER DECORATOR
import time
def timer(func):
    """ A decorator that prints how long a function took to run.end=
    Args:
        func (callable): The function being decorated.end=
    
    Returns:
        callable: The decorated function.end="""
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result
        result = func(*args, **kwargs)
        # Get the total time it toot to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper
# Test it
@timer
def sleep_n_seconds(n):
    time.sleep(n)
sleep_n_seconds(5)

# MEMOIZING: the process of storing the result of a function so that the next time the function is called with the same arguments, you can just look up the answer instead of running it again
# Let's make a wrapper for that
def memoize(func):
    """ Store the results of the decorated function for fast lookup.
    """
    # Store the results in a dict that maps arguments to results
    cache = {}
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
        if (args, kwargs) not in cache: # the second time the function is run with the same arguments, this condition will evaluate to false and we will look up the return value in the dictionary
            # Call func() and store the result.
            cache[(args, kwargs)] = func(*args, **kwargs)
        return cache[(args, kwargs)]
    return wrapper
# Test it
import time
@memoize
def slow_function(a, b):
    print('Sleeping...')
    time.sleep(5)
    return a + b
slow_function(3, 7)
slow_function(3, 7)

# When is it appropiate to use a decorator?
# When you want to add some common bit of code to multiple functions

# Create a decorator that returns the type of the returned value of a function
def print_return_type(func):
  # Define wrapper(), the decorated function
  def wrapper(*args, **kwargs):
    # Call the function being decorated
    result = func(*args, **kwargs)
    print('{}() returned type {}'.format(
      func.__name__, type(result)
    ))
    return result
  # Return the decorated function
  return wrapper
# Test it
@print_return_type
def foo(value):
  return value
print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))

# Create a decorator to count the number of times a function is used
def counter(func):
  def wrapper(*args, **kwargs):
    wrapper.count += 1
    # Call the function being decorated and return the result
    return func(*args, **kwargs)
  wrapper.count = 0
  # Return the new decorated function
  return wrapper
# Decorate foo() with the counter() decorator
@counter
def foo():
  print('calling foo()')
foo()
foo()
print('foo() was called {} times.'.format(foo.count))

# DECORATOR AND METADATA
# One of the problems of decorators is that they obscure the decorated function's metadata
# Lets explain this using the timer decorator and a slee_n_seconds function
import time
def timer(func):
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result
        result = func(*args, **kwargs)
        # Get the total time it toot to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper
# Sleep n seconds without the decorator
def sleep_n_seconds(n=10):
    """ Pause processing for n seconds.
    """
    time.sleep(n)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)
print(sleep_n_seconds.__doc__)
# Now using the decorator
@timer
def sleep_n_seconds(n=10):
    """ Pause processing for n seconds.
    """
    time.sleep(n)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)
print(sleep_n_seconds.__doc__)
# After decorating the function, we lost its default arguments and docstring attributes, and its function name is "wrapper"
# This is because the decorator overwrote the original function, and so when claling these attributes we are actually referencing the nested funcion inside the decorator
# HOW TO FIX THIS:
from functools import wraps
import time
def timer(func):
    # Decorate the nested function with wraps()
    @wraps(func) # this decorator takes the function that it is decorating as an argument
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result
        result = func(*args, **kwargs)
        # Get the total time it toot to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper
@timer
def sleep_n_seconds(n=10):
    """ Pause processing for n seconds.
    """
    time.sleep(n)
print(sleep_n_seconds.__name__)
print(sleep_n_seconds.__defaults__)
print(sleep_n_seconds.__doc__)
# warps() also gives you access to the origina, undecorated function
print(sleep_n_seconds.__wrapped__) # (keep in mind that you've always had access to this via the closure, but this is easier)

# DECORATORS THAT TAKE ARGUMENTS
# We need to add another level of function nesting
# Create a decorator that let's you specify the number of times that you want a function to run
# First we will create a function that returns a decorator (rather than a function that is a decorator)
def run_n_times(n):
    """Defines and returns a decorator"""
    def decorator(func): # this ('decorator') is the function that will be acting as a decorator
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator
# Test it
@run_n_times(3) # here we are calling the function run_n_times and then decorating my_func with the result of that function, which is a decorator
def my_func():
    print('Hello')
my_func()
# It would be the same as doing this
run_three_times = run_n_times(3)
@run_three_times
def my_func():
    print('Hello')
my_func()

# TIMEOUT: raise an error if a function runs for more than certain amount of time
import signal
def raise_timeout(*args, **kwargs): # This function simply raises a timeout error when it is called
    raise TimeoutError()
# When an 'alarm' signal goes off, call raise_timeout()
signal.signal(signalnum = signal.SIGALRM, handler = raise_timeout) # tells Python, when you see the signal whose number is 'signalnum', call the handler function, which in this case is raise_timeout
# Set off an alarm in 5 seconds in the future
signal.alarm(5)
# Cancel the alarm)
signal.alarm(5)
# Create a decorator that times out in exactly 5 seconds, and then we will improve this by creating a decorator that takes the time as an argument
def timeout_in_5s(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set an alarm for 5 seconds in the future
        signal.alarm(5)
        try:
            # Call the decorated function
            return func(*arg, *kwargs)
        finally: # we make sure that the alarm either rings (because func took more than 5s) or it gets cancelled
            # Cancel alarm
            signal.alarm(0) 
    return wrapper
# Test it with a function that definitely times out
@timeout_in_5s
def foo():
    time.sleep(6)
    print('foo!')
foo()
# Now lets make the function that creates decorators
def timeout(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            signal.alarm(n_seconds)
            try:
                return func(*arg, *kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator
@timeout(10)
def foo():
    time.sleep(6)
    print('foo!')
foo()

# Create a function 'returns' that takes a data type as argument and returns a decorator that checks whether the decorated function returns that data type
def returns(return_type):
  def decorator(func):
    def wrapper(*args, **kwargs):
      result = func(*args, **kwargs)
      assert type(result) == return_type
      return result
    return wrapper
  return decorator
  
@returns(dict)
def foo(value):
  return value

try:
  print(foo([1,2,3]))
except AssertionError:
  print('foo() did not return a dict!')

#%% EXPLORATORY DATA ANALYSIS IN PYTHON

# PROBABILITY MASS FUNCTION: like a histogram but without using bins, it puts a bar for every unique value
import empiricaldist
Pmf(educ, normalize=False)
Pmf(educ, normalize=True)
Pmf(educ, normalize=False).bar(label='educ')
plt.show()

# CUMULATIVE DISTRIBUTION FUNCTIONS
import empiricaldist
cdf = Cdf(df['col'])
cdf.plot()
plt.show()
cdf(q)
cdf.inverse(0.75) # 75th percentile

# PROBABILITY DENSITY FUNCTIONS
# Create a random variable with normal distribution
import numpy as np
sample = np.random.normal(size=1000)
# Create a normal cumulative distribution function
from scipy.stats import norm
xs = np.linspace(-3, 3) # equally spaced points from -3 to 3
ys = norm(0, 1).cdf(xs) #create a standard normal distribution and calculate the cdf
# Plot both to compare
Cdf(sample).plot()
plt.plot(xs, ys, color='gray')
plt.show()
# PDF / bell curve
xs = np.linspace(-3, 3)
ys = norm(0, 1).pdf(xs)
plt.plot(xs, ys, color='gray')
plt.show()
# However, if you plot the pdf of sample, it is a flat line because the probability of unique unique value is always approximating 0
# To go from a PMS to a PDF we can use a kernel density estimation
import seaborn as sns
sns.kdeplot(sample)
plt.plot(xs, ys, color='gray')
plt.show()

# SIMPLE LINEAR REGRESSION
from scipy.stats import linregress
res = linregress(x, y) # it does not handle NaNs
# Plot the line of best fit
fx = np.array([x.min(), x.max()])
fy = res.intercept + res.slope*fx
plt.plot(fx, fy, '-')
plt.show()

# MULTIPLE LINEAR REGRESSION
import statsmodels.formula.api as smf
resutls = smf.ols('VAR1 ~ VAR2 + VAR3', data=df).fit()
results.params

# Use a categorical variable in a regression formula
resutls = smf.ols('VAR1 ~ VAR2 + C(VAR3)', data=df).fit()

# VISUALIZING REGRESSION RESULTS
import statsmodels.formula.api as smf
model = smf.ols('VAR1 ~ VAR2 + VAR3', data=df)
results = model.fit()
results.params
# Generating predictions
df = pd.DataFrame()
df['VAR2'] = np.linspace(df['VAR2'].min(), df['VAR2'].max())
df['VAR3'] = 12 # hold this variable constant # i think this should be a list
pred12 = results.predict(df)
plt.plot(df['VAR3'], pred12, label='VAR3 = 12')
# here you could also plot the original scatter plot
plt.show()

# LOGISTIC REGRESSION
formula = 'binary_var ~ VAR1 + VAR2 + C(sex)'
results = smf.logit(formula, data=gss).fit()
results.params
# Generate a prediction
df = pd.DataFrame()
df['VAR2'] = np.linspace(df['VAR2'].min(), df['VAR2'].max())
df['VAR3'] = 12
df['sex'] = 1
pred = results.predict(df) 
plt.plot(df['VAR3'], pred12, label='VAR3 = 12')


#%% INTRODUCTION TO REGRESSION WITH STATSMODELS

# statsmodel is optimized for INSIGHT into data, while sickit-learn for PREDICTION
from statsmodels.formula.api import ols

## Ch1: Simple Linear Regression Modeling

# Estimate an OLS model
from statsmodels.formula.api import ols
model = ols("dep_var ~ indep_var", data=df)
model = model.fit()
print(model.params)

# Model without an intercept
model = ols("dep_var ~ indep_var1 + 0", data=df).fit()

# Categorical explanatory variables
model = ols("dep_var ~ indep_var1 + cat_var", data=df).fit()
print(model.params) 

## Ch2: Predictions and model objects

# To make a prediction outside the sample you need to set values for the explanatory variables
explanatory_data = pd.DataFrame({"indep_var": np.arange(20,31)}) # here we pass a vector of values for the X variable
print(model.predict(explanatory_data)) # if you use no argument, then the default is the observations used to estimate the model (here 'df')
# Or using only one value
explanatory_data = pd.DataFrame({"indep_var": [20]})
# It is better to end up with a data frame where one column has the values for the explanatory variable, and another column the predicted values
prediction_data = explanatory_data.assign(predicted_col=model.predict(explanatory_data))

# Show predicted data on a scatterplot
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure()
sns.regplot(x='indep_var', y='dep_var', ci=None, data=df)
sns.scatterplot(x='indep_var', y='dep_var', data=df, color='red', marker='s')
plt.show()

# Attributes of model objects
print(model.params) # extract parameters from a model object
print(model.params[0]) # extract intercept, if it has one
print(model.fittedvalues) # Extract fitted values of a model / Make a prediction with the original dataset
print(model.resid) # extract model residuals
print(model.summary()) # model summary
print(model.rsquared) # R squared
print(model.mse_resid) # Mean square error = residual standard error **2

#anually calculate the MEAN SQUARE ERROR, the RESIDUAL STANDARD ERROR and the ROOT MEAN SQUARE ERROR
residuals_sq = model.resid**2
resid_sum_of_sq = sum(residuals_sq)
deg_freedom = len(df.index) - len(model.params)-1 # number of observations minus number of coefficients, beware if not intercept
rse = np.sqrt(resid_sum_of_sq/deg_freedom)
rmse = np.sqrt(resid_sum_of_sq/len(df.index))

# Transform the data: square roots can be a useful transformation when the data has a right.skewed distribution

## Ch3: Assessing model fit

# Visualizing model fit: various graphs
# A good indication is that the residuals are normally distributed with mean zero
# RESIDUALS vs FITTED VALUES
sns.residplot(x='indep_var', y='dep_var', data=df, lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()
# If the dots or the lowess trend line is close to the y=0 line, then the mean of the residuals is close to zero

# Q-Q PLOT: check if the residuals follow a normal distribution (if they are close to the 45º line)
from statsmodels.api import qqplot
qqplot(data=model.resid, fit=True, line='45')
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')

# SCALE-LOCATION PLOT: plots the fitted values against the square root of the standardize residuals
# It shows whether the size of the residuals changes with the fitted values
normalized_residuals = model.get_influence().resid_studentized_internal
abs_sqrt_normalized_residuals = np.sqrt(np.abs(normalized_residuals))
sns.regplot(x=model.fittedvalues, y=abs_sqrt_normalized_residuals, ci=None, lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Sqrt of abs val of standardized residuals')

# Finding OUTLIERS
# Extreme X or Y values, or values that are far away form the regression line
# LEVERAGE: a measure of how extreme the explanatory variable values are
# INFLUENCE: a measure of how much the model would change if you do not include the observation (it depends on the leverage and the size of the residual)
# Retrieve leverage and influence
summary = model.get_influence().summary_frame() 
df['leverage'] = summary['hat_diag'] # leverage is "hat_diag"
# For influence we use cook's distance
df['cook_dist'] = summary['cooks_d']
# For example, plot how the slope of the regression line changes when dropping the observation with the highest influence
df.sort_values('cook_dist', ascending=False, inplace=True)
df2 = df.drop(df.head(1).index)
model2 = ols('dep_var ~ indep_var', data=df2)
sns.regplot(x='indep_var', x='dep_var', data=df, ci=None, line_kws={'color': 'green'})
sns.regplot(x='indep_var', x='dep_var', data=df2, ci=None, line_kws={'color': 'red'})
plt.show()

## Ch4: Simple Logistic Regression Modeling
from statsmodels.formula.api import logit
logit_model = logit('dep_var ~ indep_var', data=df).fit()
# Plot results
sns.regplot(x='indep_var', y='dep_var', data=df, ci=None, logistic=True) # 'logistic' means plot the logistic regression line
# Make predictions
explanatory_data = pd.DataFrame('indep_var': np.arange(-2, 5, 8))
prediction_data = explanatory_data.assign(prediction_value = model.predict(explanatory_data))
# Getting the most likely outcome (setting the prediction equal to 1 if probability is greater than 0,5, and 0 otherwise)
prediction_data['prediction_result'] = np.round(prediction_data['prediction_value'])
# ODDS RATIO: the probability of something happening divided by the probability of it not happening
# Calculating the odds ratio
prediction_data['odds_ratio'] = prediction_data['prediction_value'] / (1 - prediction_data['prediction_value'])
# Visualizing the odds ratio
sns.lineplot(x='indep_var', y='odds_ratio', data=prediction_data)
plt.axline(y=1, linestyle='dotted')
plt.show()
# alternatively use a log scale
plt.yscale('log')
# LOG ODDS RATIO (or 'logit')
prediction_data['log_odds_ratio'] = np.log(prediction_data['odds_ratio'])

# Assess the performance of the logistic model with a CONFUSION MATRIX: a table with two rows (actual false and actual true) and two columns (predicted false and predicted true)
# Create a confusion matrix manually
actual_response = df['dep_var']
predicted_response = np.round(mode.predict())
outcomes = pd.DataFrame(
    {'actual_response': actual_response,
    'predicted_response': predicted_response}
)
print(outcomes.value_counts(sort=False)) # the matrix
# Create a confusion matrix
conf_matrix = model.pred_table() # true negative top left, false positive top right, false negative bottom left, true positive bottom right
# Plot the confusion matrix
from statsmodels.graphics.mosaicplot import mosaic
mosaic(conf_matrix)
# Model ACCURACY: the proportion of correct predictions (true positives + true negatives / total number of observations)
TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]
acc = (TN + TP) / (TN + TP + FN + FP)
# SENSITIVITY: the proportion of true positives, in other words, the proportion of true values that were correctly predicted as true (true positives / (false negatives + true positives) )
sens = TP / (TP + FN)
# SPECIFICITY: the proportion of true negatives, in other words, the proportion of false values that were correctly predicted as false (true negatives / (true negatives + false positives) )
spec = TN / (TN + FP)
# Trade-off: increasing sensitivity will decrease specificity, and viceversa

#%% SAMPLING

# Ch1: INTRODUCTION TO SAMPLING

# Take a random sample from a data frame or series
np.random.seed(123)
df.sample(n=10) # by default there is no replacement
df['col'].sample(n=10)
df.sample(frac=0.1) # 10% of the df
# without having to declare the seed beforehand
df.sample(n=10, random_state=123)

# CONVENIENCE SAMPLING: collecting data by the easiest method (may create sample bias)

# PSEUDO-RANDOM NUMBER GENERATION
# To generate true random numbers you usually have to rely on a phisical process, but this may be slow and/or expensive
# Pseudo-random number generation is cheap and fast. It is calculated from a previous 'random' number called the seed

# Genrate random numbers with Numpy
import numpy as np
np.random.seed(123)
numpy.random.beta(a=2, b=2, size=1000) # or binomial, chisquare, t, Poisson, normal, etc
    


# Ch2: SAMPLING METHODS

# SIMPLE RANDOM SAMPLING: randomply pick one at a time, where eahc observations has equal chances of getting selected

# SYSTEMATIC SAMPLING: samples the population at regular intervals (eg, every fifth row)
# Get every Nth row from a data frame
sample_size = 100
interval = len(df) // sample_size
df.iloc[::interval]
# To make sure that systematic sample is not biased, one may plot the relationship between the value of a variable and the order of the rows
df = df.reset_index() # turn index into a column
df.plot(x='index', y='var', kind='scatter')
plt.show() # the graph should look like noise
# Randomize the order of the rows of a data frame
shuffled = df.sample(frac=1, random_state=123) # frac lets you specify the proportion of the df you want to return
shuffled = shuffled.reset_index(drop=True).reset_index() # reset the order of the index (first drop the index, then create a new one)
df.plot(x='index', y='var', kind='scatter')
plt.show()
# But once you've shuffled the rows, systematic sampling is the same as random sampling

# STRATIFIED RANDOM SAMPLING
# - PROPORTIONAL: getting an equal proportion of observations from different groups
# (eg if in the population 10% of the observations are "blue", then we would like the sample to have 10% of 'blue' observations)
stratified_sample_prop = df.groupby('var_to_stratify_by').sample(frac=0.1, random_state=123) # here you get 10% of each group
# - EQUAL COUNTS: getting the same number of observations for each group
stratified_sample_eqcount = df.groupby('var_to_stratify_by').sample(n=15, random_state=123) # here you get 15 obs from each group
# - WEIGHTED RANDOM SAMPLIG: specify weights to adjust the relative probabilty of a row being sample according to the group it belongs to
import numpy as np
df_weight = df
condition = df['col'] == True
df_weight = np.where(condition, 2, 1) # here you specify that the observations where col equals True will have two times the probability of getting picked than when col equals False. You create a column with the weight for each row
df_weight = df_weight.sample(frac=0.1, weights='weights')

# CLUSTER SAMPLING: it involves using random sampling to pick some subgroups, and then use random sampling only on those subgroups
# Cheaper that stratified samplw where you use data on all subgroups
# Randomly select K groups
import random
subgroups = random.sample(list(df['group_col'].unique()), k=3) # get 3 random values for col to use as subgroups
condition = df['col'].isin(subgroups)
clusters = df[condition]
clusters['col'] = clusters['col'].cat.remove_unused_categories() # make sure that rows with 0s are removed
clusters.groupby('col').sample(n=5, random_state=123)
# In this case we only used two levels, but we could go on (like making clusters by province, municipality, and gender, for example)


# Ch3: SAMPLING DISTRIBUTIONS

# RELATIVE ERROR of point estimates: the absolute difference between the population parameter and the point estimate
pop_mean = df['col'].mean()
sample_mean = df.sample(n=100)['col'].mean()
rel_error_pct = 100 * abs(pop_mean - sample_mean) / pop_mean

# Creating a SAMPLING DISTRIBUTION (a distribution of point estimates)
means = []
for _ in range(1000):
    means.append(df.sample(n=30)['col'].mean())
import matplotlib.pyplot as plt
plt.hist(means, bins=30)
plt.show()

# Exact sampling distributions
# Create a data frame with all possible combinations between values (here the possible values of 4 dices)
import pandas as pd
dices = expand_grid(
    {'dice1': [1,2,3,4,5,6],
     'dice2': [1,2,3,4,5,6],
     'dice3': [1,2,3,4,5,6],
     'dice4': [1,2,3,4,5,6]
     }
)
# Calculate mean of rows
dices['mean_roll'] = (dices['dice1'] + dices['dice2'] + dices['dice3'] + dices['dice4'])/4
# Barplot
dices['mean_roll'] = dices['mean_roll'].astype('category')
dices['mean_roll'].value_counts(sort=False).plot(kind='bar') # sort False is so that the x axis ranges from 1 to 6 and is not ordered by frequency
# If you increase the number of dices, the nmber of possible combinations grows exponentially, so it becomes computationally impossible to calculate exact sampling distribuions. Therefore we can make use of approximate sampling distributions

# APPROXIMATE SAMPLING DISTRIBUTIONS
import numpy as np
sample_means_1000 = []
for i in range(1000):
    sample_means.append(
        np.random.choice(list(range(1,7)), size=4, replace=True).mean() # Generate a sample mean of 4 dice rolls (with replacement)
    )
hist(sample_means_1000, bins=20)
     
# Pick numbers randomly from a list
import numpy as np
number_list = [1,5,71,66,143]
n = 3
np.random.choice(number_list, size=n, replace=False)

# STANDARD ERORR: the standard deviation of the sampling distribution


# Ch4: BOOTSTRAP DISTRIBUTIONS

# SAMPLING WITH REPLACEMENT
df.reset_index(inplace=True)
resample = df.sample(frac=1, replace=True)
# count the values of the index column to see how many were repeated
df['index'].value_counts()

# BOOTSTRAPPING: building a theoretical population from a sample
# 1) Make a resample of the same size as the original sample
# 2) Calculate the statistic of interest for this bootstrap sample
# 3) Repeat steps  and 2 many times
# The resulting statistics are bootstrap statistics and they form a bootstrap distribution
import numpy as np
bootstrap_statistics = []
for i in range(1000):
    bootstrap_statistics.append(
        np.mean(df.sample(frac=1, replace=True)['col'])
    )
import matplotlib.pyplot as plt
plt.hist(bootstrap_statistics)
plt.show()
# Beware: bootstrapping cannot correct potential biases between the chosen sample and the population. So if the sample is biased, the bootstrapped mean, for example, will not approximate the population mean very well

# The standard deviation of the bootstrapped means approximantes the standard error (the sd of the sample mean). Therefore we can use this ESTIMATED STANDARD ERROR to approximate the standard deviation of the population
Se = np.std(bootstrap_statistics, ddof=1)
Sd = Se * np.sqrt(n) # n is the original sample size

# Calculate CONFIDENCE INTERVALS of a sample statistic: 
# Two ways of calculating it:
# One is by using the mean of the bootstrapped distribution and adding and substractic X standard dviations of this distribution
mean_of_bootstrap = np.mean(bootstrap_statistics) # point estimate
lower_bound = mean_of_bootstrap - np.std(bootstrap_statistics, ddof=1)
upper_bound = mean_of_bootstrap + np.std(bootstrap_statistics, ddof=1)
print("Confidence interval of 1 SD: {} [{} - {}]".format(mean_of_bootstrap, lower_bound, upper_bound))
    # or using quantiles
lower_bound = np.quantile(bootstrap_statistics, 0.025)
upper_bound = np.quantile(bootstrap_statistics, 0.975)
print("Confidence interval of 95%: {} [{} - {}]".format(mean_of_bootstrap, lower_bound, upper_bound))
# The other way is the STANDARD ERROR METHOD which uses the INVERSE CUMULATIVE DISTRIBUTION FUNCTION
# The Bell curve is the probabilty density function
# The CDF or cumulative distribution function is the integral of the PDF
# Its inverse is the same but flipping the axis, yo you have the values of the statistic in Y axis and the accumulated probability (0 to 1) in the X axis
from scipy.stats import norm
point_estimate = np.mean(bootstrap_statistics)
std_error = np.std(bootstrap_statistics, ddof=1)
lower_bound = norm.ppf(0.025, loc=point_estimate, scale=std_error) # This assumes that the bootstrapped distribution is normal, although it is perfectly so
upper_bound = norm.ppf(0.975, loc=point_estimate, scale=std_error)


#%% HYPOTHESIS TESTING IN PYTHON

# Ch1: INTRODUCTION TO HYPOTHESIS TESTING

# Hypothesis: a statement about an unknown population parameter
#Hypothesis test: comparing two competing hypothesis
# Null hypothesis: the existing idea, assumed to be true
# Alternative hypothesis: a new "challenger" idea

# Determining whether a sample statistic if close to or far aways from an expected value
# Hypothesis: the sample mean is 100
sample_mean = df['col'].mean() # suppose it equals 90, how is this evidence for or against the hypothesis?
# As we do not have the population standard deviation, to calculate the std of the mean, we can infer it with the bootstrap distribution
import numpy as np
bootstrapped_sample_means = []
for i in range(5000):
    bootstrapped_sample_means.append(
        np.mean(df.samepl(frac=1, replace=True)['col'])
    )
std_error = np.std(bootstrapped_sample_means, ddof=1) # the std of the bootstrapped means approximates the mean std
# STANDARDIZED VALUE = (VALUE - MEAN( / STANDARD DEVIATION)
# The units are Z-SCORES
z_score = (sample_mean - 100) / std_error

# Three types of tests:
# Alt. hyp. is different from null: two-tailed
# Alt. hyp. is greater than null: right-tailed
# Alt. hyp. is less than null: left-tailed

# P-VALUE: measures the strength of support for the null hypothesis.
# It is the probability of obtaining a result, assuming that the null hypothesis is true
# Large p-values represent learge support for the null hypothesis. They mean that the obtained statistic is likely NOT in the tail of the null distribution, so the null must be true
# Small p-values are strong evidence against the null, which means that the obtained statistic is likely in the tail of the null distribution
# Calculate the p-value - Right-tailed test
from scipy.stats import norm
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)
# Calculate the p-value - Left-tailed test
from scipy.stats import norm
p_value = norm.cdf(z_score, loc=0, scale=1)
# Calculate the p-value - Two-tailed test
from scipy.stats import norm
p_value = norm.cdf(-z_score, loc=0, scale=1) + 1 - norm.cdf(z_score, loc=0, scale=1)
# or more easily
p_value = 2 * (1 - norm.cdf(z_score, loc=0, scale=1))

# STATISTICAL SIGNIFICANCE: it is a cutoff point for the p-value. It is the threshold point for rejecting the null with "beyond reasonable doubt".
# If the p-value is less than the significance level, we reject the null, otherwise fail to reject it
# To get a sense of the potential values of the population parameter, it is common to chose a confidence interval level of 1 minus the significance level: so if alpha =.05, then we use the 95% confidence interval

# TYPES OF ERRORS:
# Type 1 error: false positive, or rejecting the null when the null is correct
# Type 2 error: false positive, or not rejecting the null when the null is false


# Ch2: TWO-SAMPLE AND ANOVA TESTS

# Performing T-TESTS
# Problem: comparing sample statistics of a single variable across different groups
xbar = df.groupby('group_col')['value_vol'].mean() # Calculate means by group
# The test statistic will be the difference between the group means (in this case we assume a right_sided test where we hypothesize that the mean of one group is greater than the other)
# t = [ (sample_mean_group1 - sample_mean_group2) - (pop_mean_group1 - pop_mean_group2) ] / Standard error of "sample_mean_group1 - sample_mean_group2"
# Here the null is that the means are equal, so the second term in the denominator is 0
# Standard error of the difference of means = square root of [ sample_variance_group1/n_group1 + sample_variance_group2/n_group2 ]
# When we approximate the SE with the sample standard deviations, we add more uncertainty to the test, so we use a t-student distribution
s = df.groupby('group_col')['value_vol'].std()
n = df.groupby('group_col')['value_vol'].count()
numerator = xbar_g1 - xbar_g2
denominator = np.sqrt(s_g1**2 / n_g1 + s_g2**2 / n_g2)
t_stat = numerator / denominator
# Calculating p-values from t-statistcs
# t-statistics follow a t-student distribution
# The t distribution is similar to the normal distribution but narrower, and it gets closer to the normal distribution as the degrees of freedom increase. In fact a normal distribution is a t-distribution with infinite degrees of freedom
# Degrees of freedom: the maximum number of logically independent values in the data sample
degrees_of_freedom = n_g1 + n_g2 - 2 # (2 is the number of the sample statistics that we know or assume)
# Set the significance level
alpha = 0.1
from scipy.stats import norm
p_val = t.cdf(t_stat, df=degrees_of_freedom)
print(p_val < alpha)

# PAIRED T-TESTS
# Used when each subject or entity is measured twice, resulting in pairs of observations. This means that the values in each pair are not independent
# Eg: starting from a dset with the % of votes for a republican candidate in 2 elections. Was the % of votes for this candidate lower in the last election than the first one? It would be a left-tailed test (% in first < % in last)
# One feature of this ds is that the the votes in the first election are paired with the votes in the second election. These magnitudes are not independent since they both refer to the same electoral zone
# This means voting patterns may emerge
# For paired analyses, rather than considering the two variables separately, we can consider a single variable of the difference
df['diff'] = df['values_g1'] - df['values_g2']
# Now the null would be that diff is = to 0, and the alternative less than 0
# t-statistic = (x_diff - mu_diff) / sqr_root (S_diff^2 / n_diff)
#degrees of freedom = n_diff - 1
xbar_diff = df['diff'].mean()
s_diff = df['diff'].std()
n_diff = len(df)
t_stat = (xbar_diff - 0) / np.sqrt(s_diff**2/n_diff)
d_of_f = n_diff - 1
from scipy.stats import t
p_value = t.cdf(t_stat, df=d_of_f)

# More direct calculations of T-TESTS with 'pingouin' package
import pingouin
pingouin.ttest(x=df['diff'],
                y=0, # value of the Null
                alternative='less') # here, left-tailed test, default is “two-sided”
                
# A variation that does not require creating the diff column
pingouin.ttest(x=df['values_g1'],
                y=df['values_g2'],
                paired=True,
                alternative='less')

# ANOVA TESTS
# A test for differences between groups, where the null is 
# Eg. 'Is mean annual compensattion different for different levels of job satisfaction?'
alpha = 0.2
import pingouin
pingouin.anova(data=df,
                dv='col_compensation', # dependent variable
                between='col_job_satisfaction') # the group variable
# the p-val is stored in the 'p-unc' column
# if p-val is less than alpha, it means that at least two of the group categories have significantly different values
# But it doesn't tell us which two categories are
#To do this we need to test on each pair of categories, considering all possible combinations between categories
pingouin.pairwise_tests(data=df, dv='col_compensation', between='col_job_satisfaction', padjust='none') # padjust stands for p-value correction
# The greater the nmber of groups, the more pais we'll have, the more tests we'll do, and the greater chance of each giving a false positive significant result
# Bonferroni correction: Apply an adjustment increasing the p-values, reducing the chance of getting a false positive
pingouin.pairwise_tests(data=df, dv='col_compensation', between='col_job_satisfaction', padjust='bonf') # padjust


# Ch3: PROPORTION TESTS

# ONE-SAMPLE PROPORTION TESTS
# Standard error of a proportion = squared_root[ hypothesized_value * (1 - hypothesized_value) / n ]
# We use a Z distribution because to calculate the test statistic we only use the sample statistic and the hypothesized value, we do not make any assumptions about the population standard deviation
p_hat = (df['boolean_col'] == 1).mean()
n = len(df)
p_0 = 0.5 # hypothesized value
import numpy as np
numerator = p_hat - p_0
denominator = np.sqrt(p_0 * (1-p_0) / n)
z_score = numerator / denominator
from scipy.stats import norm
p_value = 2 * (1 - norm.cdf(z_score, loc=0, scale=1))

# TWO-SAMPLE PROPORTION TESTS
# Now we will tests a difference in proportions of the same variable taken from two different populations
# z_core = [ (p_hat_g1 - p_hat_g2) - 0 ] / SE_of(p_hat_g1 - p_hat_g2)
# SE_of(p_hat_g1 - p_hat_g2) = sqrt_of [ p_hat*(1/p_hat)/n_g1 + p_hat*(1/p_hat)/n_g2 ], where p_hat is a weighted mean of the sample proportion for each category
# p_hat = (n_g1 * p_g1 + n_g2 * p_g2) / (n_g1 + n_g2)
# so we need to calculate p_hat and n for both groups.
p_hats = df.groupby('group_col')['value_col'].value_counts(normalize=True) # value_col is boolean
ns = df.groupby('group_col')['value_col'].count()
# But you can calculate the Z-score more directly with
from statsmodel.stats.proportion import proportions_ztest
n_s = df.groupby('group_col')['value_col'].value_counts(normalize=True)
n_1s = np.array([n_s[('g1', '1')], n_s[('g2', '1')]])
n_rows = np.array([n_s[('g1', '1')] + n_s[('g1', '0')], n_s[('g2', '1')] + n_s[('g2', '0')]])
z_score, p_value = proportions_ztest(count=n_1s, nobs=n_rows, alternative='two-sided')

# CHI-TEST OF INDEPENDENCE
# The chi-square independence test compares proportions of successes of one categorical variable across the categories of another categorical variable.
# Extends propotions test to more than two groups.
# Two categorical variables are considered statistically independent when the proportion of success in the response variable is the same across categories of the explanatory variable
# You can test whether the value of the proportions of 1s in a boolean variable is not significantly different across differnt groups of a categorical variable
# For example, the null could be that the age group of a person is independent with whether or not it clicked on an ad
import pengouin 
expected, observed, stats = pingouin.chi2_independence(data=df, x='value_col', y='group_col', correction=False) # corection specifies whether to apply Yate's continuity correction, used when the sample size is very small and the degrees of freedom is 1}
print(stats)
# The order of x and y doesnt matter
# The statistic is chi squared, which quantifies how far away the observed results are form the expected values if independence was true
# Degrees of freedom = (n_g1 - 1)*(n_g2 - 1)
# You need not worry about tails or directions because it is always right-tailed as the distribution is always positive (since it is squared)
# Visualization using a PROPORTIONAL STACKED BAR PLOT
props = df.groupby('group_col')['binary_col'].value_counts(normalize=True)
wide_props = props.unstack()
wide_props.plot(kind='bar', stacked=True)
# IF the null is true, then the height of the bars should be very similar for all categories. The test determines whether the diffence in height is statistically significant

# Chi-squared GOODNESS OF FIT TEST
# Compares proportions of a single categorical variable to a hypothesized distribution
# The statistic quantifies how far the distribution of the proportions of the observed variable is from a theoretical/pre-defined distribution
# Create the table for the sampled variable
counts = df['cat_var'].value_counts()
counts = counts.rename_axis('cat_var').reset_index(name='n').sort_values('cat_var') # rename the leftmost column to cat_var, assign the counts to 'n' and sort by the var
# Create a hypothesized distribution of proportions
hypothesized = pd.DataFrame({
    'cat_var': ['cat1', 'cat2', 'cat3'], # here perhaps you could use df.cat_var.unique()
    'prop': [1/6, 1/6, 1/2, 1/6]
})
n_total = len(df)
hypothesized['n'] = hypothesized['prop']*n_total
# Visualize both distributions in a plot
import matplotlib.pyplot as plt
plt.bar(x=counts['cat_var'], y=counts['n'], color='red', lable='Observed')
plt.bar(x=hypothesized['cat_var'], y=hypothesized['n'], color='blue', lable='Hypothesized', alpha=0.5)
plt.legend()
plt.show()
# Run the test
from scipy.stats import chisquare
chisquare(f_obs=counts['n'], f_exp=hypothesized['n'])

# Ch4: NON-PARAMETRIC TESTS

# ASSUMPTIONS IN HYPOTHESIS TESTS
# 1) The samples are random subsets of larger populations
# 2) Each observations is independent (except in paired t-tests, where we correct for dependence)
# 3) The samples are large enough so that the central limit theorem applies, so that the sample distribution is normal
    # The size depends on the test.
        # One sample: 30 (for proportion tests, at least 10 successes and 10 failures)
        # Two samples/ANOVA: 30 from each group (for proportion tests, at least 10 successes and 10 failures for each group)
        # Paired: at least 30 pairs
        # Chi-squared test of proportions: 5 successes and 5 failures
# Sanity check: plot the bootstrap distribution and check if it looks normal. If it does not, then at least one assumption is not met

# What do we do if the assumptions do not hold?

# NON-PARAMETRIC HYPOTHESIS TESTS
# The tests above were all parametric tests, based on the assumption that the sampling distributions were normal and the sample size requirement is met. Non/parametric tests do not assume this, so they work better when either assumption is not met

# RANK TESTS
# Rank is the position of the value of each observation in the domain of a variable
# We can dispense with the normality assumption by performing hypothesis tests on the ranks of the numeric input

# WILCOXON SIGNED RANK TEST for paired data
# Access the ranks of a variable
from scipy.stats import rankdata
rankdata(df.col)
# Calculate the absolute difference in paired values
df['diff'] = df['col_second'] - df['col_first']
df['abs_diff'] = df['diff'].abs()
df['rank_abs_diff'] = rankdata(df['abs_diff'])
# To calculate the test statistic, you split the ranks into two groups, one for rows with negative differences and one for positive differences.
# T_minus = sum of the ranks with negative differences 
# T_plus = sum of the ranks with positive differences 
W = np.min([T_minus, T_plus])
import pingouin
pingouin.wilcoxon(x=df['col_first'], y=df['col_second'], alternative='less')

# NON/PARAMETRIC ANOVA AND UNPAIRED T-TESTS

# WILCOXON-MANN-WHITNEY TEST:  a t-test on rank data, similar to the Wilcoxon test, but works on PAIRED DATA
# First select the two columns and convert data from long to wide format
df_wide = df[['group_var', 'value_var']].pivot(columns='group_var', values='value_var')
import pingouin
pingouin.mwu(x=df_wide['group_category_1'], y=df_wide['group_category_1'], alternative='greater')

# KRUSKAL-WALLIS TEST: non-parametric ANOVA test
# in the same way that ANOVA extends t-tests to more than 2 groups, this test extends the MWU to more than 2 groups
import pingouin
pingouin.kruskal(data=df, dv='value_var', between='group_var')



# %% SUPERVISED LEARNING WITH SICKIT-LEARN

# Two types of supervised learning: classification (binary or categorical outcome) and regression (continuous target variable)

# Requirements: no missing values, data in numeric format, and stored in pandas data frames or numpy arrays

# Sintax for sickit-learn
from sklearn.module import Model
model = Model()
model.fit(X, y) # X is an array of features, y the target variable
predictions = model.predict(X_new)
print(predictions)

# It requires that the FEATURES are in an array where each column if a feature and each row is a different observation, and that the TARGET VARIABLE needs to be a single column with the same number of rows as the features
# Also, X has to have TWO dimensions at least, it cannot be a one-dimensional vector. So, for example, for a model with only one predictor, you should reshape it like this: X.reshape(-1, 1)


# Ch1: CLASSIFICATION

# K NEAREST NEIGHBOURS (KNN, mayority voting)
from sklearn.neighbors import KNeighborsClassifier
X = df[list_of_features].values
y = df['target_var'].values
print(X.shape, y.shape)
k=15 # hyperparameter
# Parameters are learned from data through training
# Hyperparameters are not learned from data and are set before training 
knn =  KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)
# Suppose X_new is an array with new observations for which we have data on their features
print(X_new.shape) # it has to match the dimensions of X
predictions = knn.predict(X_new) # here binary results are returned. By default the proportion of neighbors threshold is 0.5

# MEASURING MODEL PERFORMANCE
# ACCURACY: number of correct predictions divided by the total number of observations
# Split data into a training and testing set
from sklearn.model_selection import train_test_split
X = df[list_of_features].values
y = df['target_var'].values
# (An alternative way of defining X when you use all other columns as features)
X = df.drop("target_var", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, # we commonly use 20 to 30% of the data to test the model
                                                    random_state= 123,
                                                    stratify=y) # this is used to replicate the proportion of successes of the original variable into the train and test sets
knn =  KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test)) # return the accuracy of the model trained with train data

# Testing different values for k
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26) # try with k from 1 to 25
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_train, y_train)
#Plot results
import matplotlib.pyplot as plt
plt.figure(figsize = (8, 6))
plt.title("KNN: Varying number of nieghbors")
plt.plot(neighbors, train_accuracies.values(), label = 'Training accuracy')
plt.plot(neighbors, test_accuracies.values(), label = 'Testing accuracy')
plt.legend()
plt.xlabel('Nº of neighbors')
plt.ylabel('Accuracy')
plt.show()


# Ch2: REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
y = df['target_var'].values
X = df.drop("target_var", axis=1).values
reg = LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)
# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123, stratify=y) 
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
# Compute R-SQUARED
reg.score(X_test, y_test)

# Another assessment: MEAN SQUARED ERROR or ROOT MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred, squared=True)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# K-FOLD CROSS-VALIDATION
from sklearn.model_selection import cross_val_score, KFold
# Create the folds on which to split data into test-train
kf = KFold(n_splits = 6,
            shuffle=True, # shuffle data set before splitting it into folds
            random_state=123)
reg = LinearRegression()
cv_results =    (reg, X, y, cv=kf) # model, features, target var, KFold object
# The default score reported is R-squared
# Return an array of k cross-validation scores
print(cv_results)
# To return the root MSE, one has to ask for "negative" MSE because sickit-learn interprets that a higher score is better
cv_results = cross_val_score(reg, X, y, cv=kf, scoring='neg_mean_squared_error')
print(np.sqrt(-cv_results))

# REGULARIZATION to avoid overfitting
# Penalize large coefficientes

# RIDGE REGULARIZATION
# Loss function = OLS's loss function + alpha*sum(beta_i_squared)
# You need to choose hyperparameter alpha (where alpha=0 is equal to using OLS)
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test)) # r-squared
print(scores)

# LASSO REGULARIZATION
# Loss function = OLS's loss function + alpha*sum(abs(beta_i))
# Shrinks some coefficients to zero
from sklearn.linear_model import Lasso
for alpha in [0.1, 1, 10.0, 100.0, 1000.0]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test)) # r-squared
print(scores)
# Checking which features got selected
X = df.drop('target', axis=1).values
y = df['target']
names = df.drop('target', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
# Plot coef values
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()


# Ch3: FINE-TUNING YOUR MODEL

# CLASS IMBALANCE: a situation where one class of the target variable is much more frequent than the other class/es
# Accuracy is not always a useful metric to assess model performance. For example, in a fraud-detection model, if only 0.1% of transactions are fraudulent, then a model that predicts all transactions as legitimate would have 99.9% accuracy, so we need a different way to assess performance.

# CONFUSION MATRIX
# PRECISION/Positive predicted values: true positive over all positive predictions (eg. number of transactions correctly predicted as fraudulent over the number of transactions predicted as fraudulent (the false positives plus the true positives) ) // TP / (TP + FP)
# Pay attention to PRECISION when the priority is to minimize FALSE POSITIVES 
# RECALL / SENSITIVITY: TP / (TP + FN), high recall reflects a low false negative rate (eg. percentage of frauds correctly predicted over total number of frauds)
# Pay attention to RECALL when the priority is to minimize the number of FALSE NEGATIVES
# F1 SCORE: the harmonic mean of precision and recall, with equal weights. Equal to: precision*recall / (precision + recall). It takes into account both the number and type of errors.
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) # precision, recall, f1

# LOGISTIC REGRESSION
# Used for classification problems. Calculates the probability that an observation belongs to a binary class
# The most common strategy is to predict 1 where p>0.5 and 0 otherwise (it produces a linear decision boundary)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123, stratify=y)
logreg.fit(X_train, y_train)
logreg.predict(X_test) # here binary results are returned. By default the probability threshold is 0.5
# Return predicted probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1] # it returns probabilities for y=0 and y=1, that's why we sliced it
# Change the probability threshold
logreg.predict(X_test)

# ROC CURVE ('reciever operating characteristics)
# Visualize how different thresholds affect TP and FP rates
# Eg. when the threshold is 0, all observations will be predicted as positive, so the TP rate will be 100% and the 100%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0,1], [0,1], 'k--') # plot a dotted line from 0 to 1. It represents an AUROC of 0.5, which is what you expect if you randomly guess the class
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.xlabel('True positive rate')
plt.tilte('ROC Curve')
plt.show()
# AUC ROC: area under the ROC curve
# The perfect model would have TPR=1 and FPR=0. Therefore the best AUROC would be 1
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_probs)

# HYPERPARAMETER-TUNNING (optimize the model)
# HYPERPARAMETER: a parameter you specify before fitting a model
# Eg: choosing alpha for Ridge and Lasso, k for KNN
# Try different values, fit each model with each value, see how each model performs and choose the one which performs the best
# We can still split the data into train and test, and perform cross-validation on the training set, withholding the test data for final evaluation

# GRID SEARCH CROSS-VALIDATION: on approach for hyperparameter tunning
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=123)
parameter_grid = {'alpha': np.linspace(0.001, 1, 10),
                    'solver': ['sag', 'lsqr']} # names of hyperparameters
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, parameter_grid, cv=kf)
ridge_cv.fit(X_train, y_train) # here the CV grid search is performed
print(ridge_cv.best_params_, ridge_cv.best_score_) # retrieve the hyperparameters that perform the best along with their mean CV score
# Limitation: it doesn't scale well because the number of fits is equal to the number of hyperparameters multiplied by the number of values and by the number of folds
# Eg: 10-fold cv with 3 hyperparams and 30 values = 900 fits
# Evaluate model performance on the test set
test_score = ridge_cv.score(X_test, y_test)
print(test_score)
# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("Accuracy of logistic regression classifier: ", best_model)


# RANDOMIZED GRID SEARCH CROSS-VALIDATION: rather than exhaustively searching through all options, it picks random parameter values
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=123)
parameter_grid = {'alpha': np.linspace(0.001, 1, 10),
                    'solver': ['sag', 'lsqr']} # names of hyperparameters
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, parameter_grid, cv=kf, n_iter=2) # 'n_iter': number of hyperparameter values tested. Here it performs 10 fits (5 fold cv, with )
ridge_cv.fit(X_train, y_train) # here the CV grid search is performed
print(ridge_cv.best_params_, ridge_cv.best_score_)
# Evaluate model performance on the test set
test_score = ridge_cv.score(X_test, y_test)
print(test_score)



# Ch4: PREPROCESSING AND PIPELINES

# Preprocessing data
# Sickit-learn does not admit categorical features, so we need to conert them to numeric using dummy variables
import pandas as pd
dummies = pd.get_dummies(df['categorical_col'], drop_first=True)
df = pd.concat([df, dummies], axis=1))
df.drop('categorical_col', axis=1, inplace=True)
# If the data frame only has one categorical feature you can do this more directly
df = pd.get_dummies(df, drop_first=True)

# Reindex the columns of the test set aligning with the train set
df_train = pd.get_dummies(df_train, drop_first=True)
df_test = pd.get_dummies(df_test, drop_first=True)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

# HANDLING MISSING DATA
# Get the count of missing values for each column
print(df.isna().sum().sort_values())
# Drop them
df = df.dropna(subset=['col_with_na_1', 'col_with_na_2']) # one criterion is to remove misisng values in columns with less than 5% of missing values
# Imputating missinga values (with the mean, median, mode (for categorical vars))
# !!! Data leakage: split data BEFORE imputing to avoid leaking test information to the model
from sklearn.impute import SimpleImputer
X_cat = df['categorical_col'].values.reshape(-1,1) # categorical features
X_num = df.drop(['categorical_col', 'target_var'], axis=1).values # numerical features
y = df['target_var'].values
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size = 0.3, random_state= 123)
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size = 0.3, random_state= 123)
# Impute missings in categorical var
impute_categorical = SimpleImputer(strategy='most_frequent') # we use the mode of the categorical variable to replace missing values
# By default, SimpleImputer identifies missing values as np.Nan
# SimpleImputer is a transformer
X_train_cat = impute_categorical.fit_transform(X_train_cat) # imputation is done here
X_test_cat = impute_categorical.transform(X_test_cat)
# Impute missings in numeric vars
impute_numeric = SimpleImputer(strategy='mean') # mean is the default
X_train_num = impute_numeric.fit_transform(X_train_num) # imputation is done here
X_test_num = impute_numeric.transform(X_test_num)
# Re-join feautres
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

# PIPELINE for imputation and model-fitting
# In this example we perform a binary classification
from sklearn.pipeline import Pipeline
# Drop some rows misisng values
df = df.dropna(subset=['col_with_na_1', 'col_with_na_2']) 
X = df.drop('cat_var', axis=1).values
y = df['cat_var'].values
# Build a pipeline by specifying the steps in tuples, with their names and functions
# Each step, except for the last, must be a transformer
steps = [("imputation", SimpleImputer()),
            ('logistic_regression', LogisticRegression())]
pipeline = Pipeline(Steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123, stratify=y)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

# CENTERING AND SCALING DATA
# Many ML models use some form of distance measure (eg. KNN), so if we have features on far larger scales, they can disproportionately influence the model
# Therefore we want to have features on a similar scale. To do that we have to normalize or standardize our data scaling and centering)
# There are several ways to do this:
# - STANDARDIZATION: substract the mean and divide by the variance. All features will be centered around zero and have variance equal to one
normalized_value = (value - min_value) / (variance_value) # we will use sickit-learn
# - 2: substract the minimum and divide by the maximum. All features will have a range from 0 to 1
normalized_value = (value - min_value) / (max_value)
# - NORMALIZATION: so that the data ranges from -1 to 1
normalized_value = (value - min_value) / (max_value - min_value) * 2 - 1
# Example: standardization
from sklearn.preprocessing import StandardScaler
X = df.drop('cat_var', axis=1).values
y = df['cat_var'].values
# !!! split data BEFORE scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Verify changes
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
# Or you can put the scaler in the pipeline
from sklearn.pipeline import Pipeline
steps = [('scaler', StandardScaler())
            ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))

# Another way to scale: MIN-MAX SCALER
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)) # set the range of the variable from 0 to 1
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

# Other examples are MaxAbsScaler and Normalizer

# Now add CROSS-VALIDATION to a pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler())
            ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)} # when passing only one parameter without names separately, the name of the key always has to be "modelname__paramname"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 123)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(cv.best_score_, cv.best_params_)
### Another example
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Create steps
steps = [("imp_mean", SimpleImputer()), 
         ("scaler", StandardScaler()), 
         ("logreg", LogisticRegression())]
# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}
# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)
# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_, tuning.score(X_test, y_test)))

# EVALUATING MULTIPLE MODELS
# Guiding principles:
# - Size of dataset: fewer feautres -> simpler model. Some models like neural netwroks require a lot of data to perform well.
# - Interpretability: some models can be explained more easily which can be important for stakeholders (eg linear regression)
# - Flexibility: may improve accuracy by making fewer assumptions about the data (eg. KNN which does not assume a linear relationship between y and X)
# Regression models can be valuated with RMSE and R2
# Classification models can be evaluated with accuracy, confusion matrix, precision, recall, f1-score,  ROC AUC

# EVALUATING MULTIPLE MODELS - CLASSIFICATION
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X = df.drop('cat_var', axis=1).values
y = df['cat_var'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 123)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {'LogisticRegression': LogisticRegression(), 'KNN':KNeighborsClassifier(), 'Decision Tree':DecisionTreeClassifier}
results = []
for model in models.values():
    kf = KFold(n_splits=6, shuffle=True, random_state=123)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
# Evaluate on the test set
for name,  model in models.items():
    model.fit(X_train_scaled, y_test)
    test_score = model.score(X_test_scaled. y_test)
    print('{} test set accuracy: {}'.format(name, test_score))

# EVALUATING MULTIPLE MODELS - REGRESSION
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []
# Loop through the models' values
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  # Perform cross-validation
  cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
  # Append the results
  results.append(cv_scores)
# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()
# Evaluate with RMSE
for name, model in models.items():
    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    # Calculate the test_rmse
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("{} Test Set RMSE: {}".format(name, test_rmse))


#%% UNSUPERVISED LEARNING IN PYTHON

# Pure patern discovery, unguided by a prediction task


# Ch1: CLUSTERING FOR DATA SET EXPLORATION

# K-MEANS CLUSTERING
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples) # samples has to be an n-dimensional numpy array where columns are features and rows are feature values
labels = model.predict(samples)
# More direct method
labels = model.fit_predict(samples)
# If you get new observations, the model can assign them to a cluster without being re-fitted (the model remembers the centroid of each cluster)
new_labels = model.predict(new_samples)

# Get the centroids of the clusters
centroids = model.cluster_centers_

# Scatter plot colored by class
import matplotlib.pyplot as plt
xs = samples[:,0] # get coordinates
ys = samples[:,1]
plt.scatter(xs, ys, c=labels)
plt.show()

# Add the centroids to the scatter plot
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,1]
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(xs, ys, c=labels, alpha=0.5)
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()

# EVALUATING A CLUSTERING
# The simplest example is when you have a variable that identifies each observation with a category
import pandas as pd
df = pd.DataFrame({'cluster_label': labels, 'real_category':species})
pd.crosstab(df['cluster_label'], df['real_category'])
# But in most cases you do not have this variable
# A good clustering has "tight" clusters, meaning they are not very spread out
# INERTIA measures how spread out the clusters are (lower is better)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples) # when the fit measure is called inertia is measured automatically
print(model.inertia_)
# K-means always minimizes inertia, having more clusters lowers the inertia
# Which is the best number of cluster to choose? It is a trade-off between having tight groups and few groups
# Usually the "elbow" if the INERITA PLOT is chosen, where inertia begins to decrease more slowly
# Plot an INERTIA PLOT
ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(samples)
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# TRANFORMING FEATURES TO IMPROVE CLUSTERING
# When the features have very different variances, clustering usually does not work very well
# The greater the variance of a feature, the greater its influence will be in the model
# So we can transform the data so that the features have equal variance: SCALER
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaler = scaler.transform(samples)
# Now do it in a PIPELINE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, model)
pipeline.fit(samples)
labels = pipeline.predict(samples)


# Ch2: VISUALIZATION WITH HIERARCHICAL CLUSTERING AND T-SNE

# HIERARCHICAL CLUSTERING
# One type is called "AGGLOMERATIVE":
# - Every country begins in a separate cluster
# - At each step, the two closest clusters are merged
# - It continues until all clusters are merged in a single cluster
# "DIVISIVE" hierarchical clustering works the other way around

# AGGLOMERATIVE HIERARCHICAL CLUSTERING
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method='complete') # 'complete' linkage: the dist between clusters is equal the max distance between their samples
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()

# Get cluster labels in hierarchical clustering
# Clusters at any intermediate level can be obtained
# The y-axis of the dendrogram shows the distance between merging clusters (the clusters that werwe merged at that height)
# The distance between clusters is measured by a LINKAGE METHOD
# Extracting cluster labels:
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
mergings = linkage(samples, method='complete')
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
labels = fcluster   
import pandas as pd
df = pd.DataFrame({'cluster_label': labels, 'real_category': cats})
pairs = pd.crosstab(df['cluster_label'], df['real_category'])
print(pairs.sort_values('cluster_label'))

# t-SNE: 't-distributed stochastic neighbor embedding'
# It maps samples from a higher-dimensional space to a 2D or 3D space
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100) # usually you need to try several values before getting the learning rate right, try between 50 and 200
transformed = model.fit_transform(samples) # it does not have separte fit() and transform() methods, so you cant extend the map to include new data
xs = transformed[:, 0]
xs = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()
# The axes do not have any interpretable meaning, and in fact they are different every time it is applied


# Ch3: DECORRELATING YOUR DATA AND DIMENSION REDUCTION

# DIMENSION REDUCTION: find patterns in data and uses these patterns to re-express it in a compressed form
# This makes subsequent computation much more efficient
# The most important function is to reduce a data set to its 'bare bones', discarding noisy features that cause problems for supervised learning tasks

# PRINCIPAL COMPONENT ANALYSIS
# Performs dimension reduction in two steps:
# - 1º DECORRELATION: rotates the observations so that they are alligned with the axes, and it shifts them so they have mean zero
# it does not change the dimension of the data
# - 2º DIMENSION REDUCTION: 
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples) # an n-dimensional numpy array
transformed = model.transform(samples) # returns a new array with the same number of rows and columns as the original. Rows are samples and columns are PCA features
# The PCA features are NOT linearly (Pearson) correlated
# The PIRNCIPAL COMPONENTS of the data are the directions on which the samples vary the most (or 'directions of variance'). They are the ones allgined with the coordinate axes
# After the model is fitted, the PCs are available
print(model.components_)
# Check this graphically
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)
print(correlation)

# INTRINSIC DIMENSION
# The number of features required to approximate a data set without losing much information
# Eg: in a data set with coordinates of a flight trayectory, in fact you only need the feature 'displacement along the flight path' to describe the data, so it is intrinsically one-dimensional
# It tells us how much a data set can be compressed
# It can be identified with PCA: the number of PCA features that have a high variance equals the intrinsic dimension
# Plot the variances of PCA features
model = PCA()
model.fit(samples)
features = range(model.n_components_)
plt.bar(features, model.explained_variance_)
plt.xticks(features)
plt.xlabel('PCA Feature')
plt.ylable('variance')
plt.show()
# !!! It is not always unambiguous

# Plot the direction of the first principal component
# Make a scatter plot of the untransformed points
plt.scatter(samples[:,0], samples[:,1])
# Create a PCA instance: model
model = PCA()
# Fit model to points
model.fit(grains)
# Get the mean of the grain samples: mean
mean = model.mean_
# Get the first principal component: first_pc
first_pc = model.components_[0,:]
# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
# Keep axes on same scale
plt.axis('equal')
plt.show()

# DIMENSION REDUCTION WITH PCA: discard low variance features
from sklearn.decomposition import PCA
# First specify how many features to keep (a good choice would be the intrinsic dimension of the data set)
model = PCA(2)
model.fit(samples)
transformed = model.transform(samples)
print(transformed.shape) # only two features

# PCA assumes that the high variance features are informative (there are cases in which it does not hold)
# In cases where it doesn't, you need to change some things
# Eg: a data set where rows are documents and columns are all words in all documents, and cells are the number of times each word appears in each document
    # Many features will have most rows as zeroes because they are words that appear only in certain documents
    # These are called "sparse" features and they can be represented in a special type of array called csr scipy.sparse.csr_matrix instead of a numpy array
    # The csr_matrix only remembers non-zero entries
    # sickit learn's PCA does not support this matrices so you have to use TruncatedSVD instead
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components = 3)
model.fit(documents_df) # input csr_matrix
transformed = model.transform(documents_df)
 
# TF-ID WORD FREQUENCY ARRAY
# Create a sparse matrix where each row is a document/pieace of text and
# each column is a word that appears in at least one document,
# and the cells are the nº of times each word appears in each doc.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(documents) # 'documents' is a list with strings, where each string is a text
print(csr_mat.toarray()) # count values
# Get the words: words
words = tfidf.get_feature_names()
print(words)
# "idf": a weighting scheme that reduces the influence of frequent words (like "the")

# PIPELINE: count word frequency, perform PCA to reduce dimensionality, and separate into clusters
# Source: https://www.lateral.io/resources-blog/the-unknown-perils-of-mining-wikipedia
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd
# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)
# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)
# Create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)
pipeline.fit(articles)
# Calculate the cluster labels: labels
labels = pipeline.predict(articles)
# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})
# Display df sorted by cluster label
print(df.sort_values('label'))


# Ch4: DISCOVERING INTERPRETABLE FEATURES

# NMF: NON-NEGATIVE MATRIX FACTORIZATION
# Dimension reduction technique, but models are interpretable (unlike PCA)
# It requires all sample features to be non-negative (>=0)
# It decomposes samples as sums of their parts
    # eg. it expresses documents as combinations of topics or themes / images as combinations of common patterns
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples) # entries are always non-negative
nmf_features = model.transform(samples) # features are always non-negative
# As in PCE, the dimension of NMF's components is the same as the dimension of the samples
print(model.components_.shape)
# Reconstruct (partially) the sample from its features: multiply nmf_features by nmf_components, and adding up
# This calculation can also be expressed as a product of matrices

# Articles application: when NMF if applied to documents its components can be interpreted as topics
    # Therefore, NMF features combine topics to form documents. Topics are groupi
    # Eg. a Wikipedia article on Di Caprio may have 0.7 of 'acting' topic, 0.02 of 'climate change' topic, etc.
# When NMF is applied to im ages, the NMF components represent frequent patterns

# Represent a collection of IMAGES as a non-negative array
# Eg. greyscale where images can be encoded by the brightness of each pixel and nothing else (0 black, 1 white)
# If you order the pixels from left to right or up to bottom, you can flatten an n-dimensional array of brightness values
# Therefore if you have K images os the size/number of pixels, then you can convert them to a K-dimensional array
    # where each row contains all the pixels for each image, and each column corresponds to a pixel
# To recover the image as a grid, use
 bitmap = sample.reshape((2,3))
 print(bitmap)
 from matplotlib import pyplot as plt
 plf.imshow(bitmap, cmap='gray', interpolation='nearest')
 plt.show()
 # Or make it a function
 def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
# Visualize the components of the NMF model
from sklearn.decomposition import NMF
model = NMF(n_components=7)
features = model.fit_transform(samples)
for component in model.components_:
    show_as_image(component)
digit_features = features[0,:]
print(digit_features)

# Build a RECOMMENDATION SYSTEM using an NMF model
# Suppose you work at a newspaper and you are ask to make an algorithm that 
# recommends articles similar to the article being read 
# Strategy:
# - Apply NMF to the word frequency array of the articles ('articles')
# - Then the NMF features will be the topics
# - So similar documents will have similar NMF values.
# - But how do you compare NMF values????
# Two articles that speak about the same topics may be written in different styles
# So one would expect the word count not to be the same
# However, in a graph where each axis is a topic, these two articles will probably lie
# on the same line passing through the origin. Therefore we should compare the direction
# of the vectors formed by the topic combination. To do this we can use
# COSINE SIMILARITY which compares the angles between these lines
from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)
# Compute the cosine similarity of the articles
    # normalize the model's features
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# Select one article you want to get a recommendation for
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article) # this returns the cosine similarities
# Or you can label the similarities with the titles of the articles
import pandas as pd
df = pd.DataFrame(norm_features, index=titles)
current_article = norm_features.loc['Title 1']
similarities = df.dot(current_article)
# Now display the articles with the highest cosine similarity
print(similarities.nlargest())

#%% LINEAR CLASSIFIERS IN PYTHON

# Auxiliary: a function to plot the classified data in a region
# https://people.cs.umass.edu/~sheldon/teaching/cs335/lec/09-classification.html

# This course has a lot of graphical explanations, so it is good to watch the videos.

# A classifier is LINEAR when it has a linear DECISION BOUNDARY

# DECISION BOUNDARY: the surface separating different predicted classes
# CLASSIFICATION: supervised learning when Y has categories.
# LINEARLY SEPARABLE DATA: a data set that can be perfectly explained by
#   a linear classifier. With this kind of data you can get training accuracy
# of 100%.

# Ch1: AAPLYING LOGISTIC REGRESSION AND SVM
# (this is an introductory part, the last two chapters are about each method    )

# LOGISTIC REGRESSION
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
wine = sklearn.datasets.load_wine()
lr = LogisticRegression()
lr.fit(wine.data, wine.target)
lr.score(wine.data, wine.target) # training accuracy
# Output CONFIDENCE SCORES instead of definite predictions
lr.predict_proba(wine.data[:1])

# LINEAR SUPPORT VECTOR CLASSIFIER
import sklearn.datasets
from sklearn.svm import LinearSVC
wine = sklearn.datasets.load_wine()
svm = LinearSVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)

# NON-LINEAR SUPPORT VECTOR MACHINE
# SVC fits a non-linear SVM by default
import sklearn.datasets
from sklearn.svm import SVC
wine  sklearn.datasets.load_wine()
svm = SVC()
svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)

# Code example: Plot different SVM classifiers in the iris dataset
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

# Ch2: LOSS FUNCTIONS

# Some theory

# DOT PRODUCT:
# The dot product of vectors (1,2,3) and (4,5,6) is  1*4+2*5+3*6 = 32
# Numpy version
import numpy as np
x = np.array([1,2,3])
y = np.array([4,5,6])
np.sum(x*y)
# Python version
x@y

# LINEAR CLASSIFIER PREDCITION
# The predict method calculates the
# raw model output = coefficients * features + intercept (where * is the dot product),
# Then checks the sign of the raw model output, and predicts one class if it
#   is negative and the other if it is possitive
# This applies for LR and SVM, which have the same predict function,
# but different fit functions (this diff relates to loss functions)
# Get the COEFFICIENTS and INTERCEPT of the predicted model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)
lr.coef_ # coefficients
lr.intercept_ # intercept
# Replicate the equation above
lr.coef_ @ X[10] + lr.intercept_ # raw model output

# LOSS FUNCTIONS
# Many ML algorithms involve minimizing a loss
# The general idea is that you "try" different combinations of values for the
# coefficients until you get the minimum value for the loss function
# The loss function is a penalty score that tells us how poorly the model is
# doing in the training data. The .fit method minimizes this.

# For classification problems, the squared error is not an appropiate loss function
# It is natural to use the number of errors made. The "0-1 LOSS" function scores
# 0 is the prediction is correct, and 1 otherwise.
# Logistc regression and SVM do NOT minimize the 0-1 loss because it is
# difficult to do so (reasons beyond the course)

# MINIMIZING A LOSS FUNCTION
# Eg. minimize Y=X^2
from scipy.optimize import minimize
minimize(np.square, 0).x # the first argument is X^2, the second our initial
# guess. ".x" grabs the input value that makes the function as small as possible
# The result is 0 bcs X^2 is minimized when X=0

# LINEAR REGRESSION "FROM SCRATCH" / LINEAR REGRESSION MANUALLY
# The squared error, summed over training examples
def my_loss(w):
    s = 0
    for i in range(y.size):
        # Get the true and predicted target values for example 'i'
        y_i_true = y[i]
        y_i_pred = w@X[i]
        s = s + (y_i_true - y_i_pred)**2
    return s
# Returns the w that makes my_loss(w) smallest
w_fit = minimize(my_loss, X[0]).x
print(w_fit)
# Compare with scikit-learn's LinearRegression coefficients
lr = LinearRegression(fit_intercept=False).fit(X,y)
print(lr.coef_)


# Ch3: LOGISTIC REGRESSION AND REGULARIZATION

# Key Hyperparameters: C(inverse rgularization strength, penalty(type of regularization),
# and multi_class (type of multi-class))

# REGULARIZATION STRENGTH
# In sickit learn, the hyperparameter "C" is the inverse of the regularization
#   strength: larger C means less regularization and vice versa

# Usually, adding regularization decreases the train accuracy. This is because
#   we are modifying the loss function to penalize large coefficients, which
#   "distracts" the original goal of optimizing accuracy. But at the same time
#   it usually improves test accuracy.

# L1 regularization: LASSO
# L2 regularization: RIDGE

# Choose the type of regularization in a Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', penalty='l1') # 'solver' controls the
    # optimization method used to find the coeffients. We need to set this here
    # because the default solver is not ocmpatible with L1 regularization.

# Interpret the raw model output of the LogisticRegression classifier as a probability
# The Sigmoid function translates the raw model output (between -Inf and +Inf)
#   to a probability between 0 and 1

# MULTI-CLASS LOGISTIC REGRESSION
# Classification with more than two categories

# Approach 1: ONE-VERSUS-REST STRATEGY ('ovr')
# One approach is to train a series of binary classifiers for each class
lr0 = LogisticRegression()
lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr0.fit(X, y==0)
lr1.fit(X, y==1)
lr2.fit(X, y==2)
# Then you choose the class whose model gives the largest raw model output
lr0.decision_function(X)[0]
lr1.decision_function(X)[0]
lr2.decision_function(X)[0]
# Sickit learn has the option to do the previous automatically
lr_ovr = LogisticRegression(multi_class='ovr')
lr_ovr.fit(X, y)

# Approach 2: MULTINOMIAL / SOFTMAX / CROSS-ENTROPY LOSS
# modify the loss function so that it directly tries to optimize accuracy on
# multiclass problem.
# The difference is that you only fit a single classifier which directly
# predicts the best class. Yet, it can be more complicated. Sometimes is 
# perfoms better than the previous approach.
lr_mn = LogisticRegression(multi_class='multinomial')
lr_mn.fit(X, y)
# See the coefficients and intercepts of this model
lr_mn.coef_.shape
lr_mn.intercept_.shape
# You get one coefficient per feature, per class, and one intercept per class.
# (the same quantity as in the multiple binary classificators)


# Ch4: SUPPORT VECTOR MACHINES

# Key Hyperparameters: C (inverse rgularization strength, kernel (type of kernel),
# and gamma (inverse RBF smoothness))

# A Linear SVM if a classifier that uses the HINGE function as a loss function
#   and L2 regularization.

# Because it uses a hinge loss function (see graph), if a training sample falls
# in the region with zero loss, it does not contribute to the fit (meaning if that
# should you remove it, nothing would change). Different from Logistic Reg., in
# SVM not all samples matter to the fit.

# SUPPORT VECTOR: a traning example that is NOT in the flat part of the loss
#   diagram. It is an example that is either incorrectly classified, or very
#   close to the boundary. How you define "close" depends of the regularization
#   strength. They are the examples that matter to your fit.

# 'Max-margin' viewpoint: the SVM maximizes the margin for linearly separable
# datasets.
# Margin: distance from the boundary to the closest points.
# The idea behind this is the line (or many
# lines) that the SVM tries to draw between the different points is the one
# which is the farthest from the points. It tries to leave the largest space
# between the clouds of points; to draw a line that is halfway between them.
# This viewpoint applies to linearly separable data. For non-linearly sep. data,
# this concept can be extended.

# DEMONSTRATION: show how a SVM trained with only the support vectors is the 
# same as one trained with the whole dataset
# Train a linear SVM
svm = SVC(kernel="linear")
svm.fit(X,y)
# Make a new data set keeping only the support vectors
print("Number of original examples", len(X))
print("Number of support vectors", len(svm.support_)) # get the support vectors of a SVM
X_small = X[svm.support_]
y_small = y[svm.support_]
# Train a new SVM using only the support vectors
svm_small = SVC(kernel="linear")
svm_small.fit(X_small, y_small)

# KERNEL SVM
# When dealing with non-linearly, linear SVMs are not so useful.
# Fitting a linear model in a transformed space (like x^2), corresponds to
# fitting a non-linear model (like an elipse region) to the original space (watch the
# graphical explanation)
# In general, the transformation is not always squaring, and the boundary is not
#always an elipse
# KERNELS implement efficient transformations in a computationally-efficient way.
from sklearn.svm import SVC
svm = SVC(gamma=1) # default kernel is 'rbf'
# 'gamma' controls the smoothness of the boundary (smaller gamma = less segmented boundaries)
# With the right hyperparameters, rbf SVMs are capable of perfectly separating 
# almost any dataset. But of course large gamma leads to overfitting.

# Choose hyperparameter gamma
# Instantiate an RBF SVM
svm = SVC()
# Instantiate the GridSearchCV object and run the search
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X, y)
# Report the best parameters
print("Best CV params", searcher.best_params_)

# Choose hyperparameters gamma and C
# Instantiate an RBF SVM
svm = SVC()
# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)
# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

# Comparing SVM vs LogisticRegressions (watch video)

# SGD CLASSIFIER: 'stochastic gradient descent'
# It can handle very large data sets better than SVM and LR
# Switch between LR and linear SVM
from sklearn.linear_model import SGDClassifier
logreg = SGDClassifier(loss='log_loss', alpha=1) #'alpha' is the regularization
# parameter, equivalent to 1/C
linsvm = SGDClassifier(loss='hinge')
# Example
# We set random_state=0 for reproducibility 
linear_classifier = SGDClassifier(random_state=0)
# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['hinge', 'log_loss']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)
# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))

#%% MACHINE LEARNING WITH TREE-BASED MODELS IN PYTHON

# --- Ch1: CLASSIFICATION AND REGRESSION TREES (CARTs) --- 

# Given a labelled data set, a CART learns a sequence of if-else questions about individual features in order to infer the labels
# In contrast to linear models, they can capture non-linearities
# They DONT require feature scaling

# MAXIMUM DEPTH: the max number of branches from top to bottom
# DECISION REGION: region in a feature in space whare all instances are assigned to one class label
# DECISION BOUNDARY: surface that separates a decision region
# A classification train produces RECTANGULAR decision regions
# DECISION TREE: a data structure consisting of a hierarchy of nodes
# NODE: a point that involves a question or a prediction
# ROOT: the node at which the tree starts growing, it has no parent node, gives rise to two children node
# INTERNAL NODE: one parent node, gives rise to two children nodes 
# LEAF: a node that has no children nodes, has only one parent node, and makes a prediction
# INFORMATION GAIN: the nodes of a classification tree are grown recursively, its obtaintion depends on the state of its predecessors
# To produce the purest leaf possible at each node, a tree asks a question involving one feature F and a split point SP
# It chooses f and sp by maximazing information gain
# IG(f, sp) = I(parent) - [N_left/N*I(left) + N_right/N*I(right)]
# I() is the impurity of a node
# To measure I() you can use different criteria: Gini or entropy
# When an UNCONSTRAINED tree is trained, nodes are grown recursively
# At each non-leaf node, the data is split based on f and sp in such a way to maximize IG.
# IF the IG obtained by splitting a node is null (IG(node)=0), the node will be a leaf
# If the tree is constraied to max. depth K, all nodes having a depth of 2 will be leafs even if the IG is not nill

# CLASSIFICATION TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 123, stratify=y) 
dt = DecisionTreeClassifier(max_depth=2, random_state=123)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

#  Change the infromation criterion
dt = DecisionTreeClassifier(max_depth=2, random_state=123, criterion='entropy')
dt = DecisionTreeClassifier(max_depth=2, random_state=123, criterion='gini')

# REGRESSION TREE
# IMPURITY: I(node) = MSE(node) =  1/N_node * SUM_i_in_node[(y_i - y_hat_node)**2],     y_hat_node = 1/N_node * SUM_i_in_node(y_i)
# PREDICTION: y_pred = 1/N_leaf * SUM_i_belongs_to_leaf(y_i)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared error as MSE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 123) 
dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=0.1, random_state=123)
# 'min_samples_leaf': imposes a stopping condition so that each leaf has to contain at least 10% of the data
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_pred, y_test)
rmse_dt = mse_dt**(1/2)
print(rmse_dt)


# --- Ch2: THE BIAS-VARIANCE TRADE-OFF ---

# In supervised learning you assume that there is a mapping between features and labels: y = f(X)
# In reality, data generation is always accompanied with randomness
# The goal is to find the best model f_hat that approximates f
# The end goal is for f_hat to achieve a low predictive error on unseen data
# You may encounter two difficulties:
# - Overfitting: whne f_hat fits the noise of the training set,
    # which results in low predictive capacity on unseen data
# - Underfitting_ when f_hat is not flexible enough to approximate f

# GENERALIZATION ERROR: a measure of how much the model generalizes on usneen data
# GE(f_hat) = bias**2 + variance + irreducible error
# Bias: how much f_hat differs from f on average (high bias -> underfitting)
# Variance: how much f_hat is inconsistent over different trainin sets (high variance -> overfitting)
# Model complexity: sets the flexibility of f_hat (eg. max depth, min samples per leaf)
# The goal is to find the model complexity that achieves the lowest generalization error,
    # in a context where, as themodel becomes more complex, bias decreases and variance increases

# DIAGNOSING BIAS AND VARIANCE PROBLEMS
# The generalization error cannot be estimated directly because the truw f is unkknown,
    # you usually have only one data set, and noise is unpredictable
# A solution is to split data in train and test, fit f_hat to the training set
    # evaluate the error on the unseen test set and approximate the GE(f) with GE(f_hat)
# !!! The test set should NOT be touched until we are confident about f_hat's performance
# To obtain a reliable estimate of f_hat's performance one should use cross-validation
# Once you have computed f_hat's CV error, you have to compare it with f_hat's training
# set error. If it is, f_hat suffers from high variance, it overfits the training set
# To remedy overfitting you can try decreasing its complexity or gather more data
# If the CV error is roughly equal to the training set error, but both errors
    # are much larger than the desired error, then f_Hat underfits the data,
    # so increase complexity or gather more relevant features

# K-fold Cross Validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_State=123)
MSE_CV = (-1) * cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1) # 'n_jobs': exploit all available CPUs
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
print('Training CV MSE: {:.2f}'.format(MSE(y_train, y_pred_train)))
print('Test CV MSE: {:.2f}'.format(MSE(y_test, y_pred_test)))


# ADVANTAGES OF CARTs: simple to understand and interpret, easy to use, non-linear
    # also you don't need to pre-process your features a lot
# DISADVANTAGES OF CARTs: it can only produce orthogonal decision boundaries,
    # sensitive to small variations in the training set, CARTs suffer from high
    # variance when they are trained without constraints (overfitting)
# Solution: ENSEMBLE LEARNING

# ENSEMBLE LEARNING
# 1) Train different models on the same data set
# 2) Each model makes its predictions
# 3) Aggregate the predictions of individual models in a meta-model and output it
# It is more robust and less prone to errors
    #If one model is wrong, it is compensated by the others
# VOTING CLASSIFIER / Hard voting / mayority voting
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
lr = LogisticRegression(random_state=123)
knn = KNN()
dt = DecisionTreeClassifier(random_state=123)
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbors', knn), ('Classification Tree', dt)]
for classifier_name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('{:s} : {:3f}'.format(classifier_name, accuracy_score(y_test, y_pred)))
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(y_test)
print('Voting classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))


# --- Ch3. BAGGING AND RANDOM FOREST ---

# BAGGING: bootstrap aggregation
# A type of ensemble but where are models are the same technique
# One algorithm that estimates different models with different bootstrap samples
# This reduces the variance of individual models in the ensemble
# (it applies to all models, not only trees)
# In CLASSIFICATION the final prediction is obtained by mayority voting of the
    # different trees (BaggingClassifier in sk.learn)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=6, random_state=123)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print('Accuracy of Bagging Classifier: {.3f}'.format(accuracy))
# In REGRESSION the final prediction is the average of the predictions of the
    # different trees (BaggingRegressor in sk.learn)

# OUT OF BAG EVALUATION
# Because of bootstrapping, some instances may be sampled several times,
    # and some may not be sampled at all (on avg 63% and 37% in each tree)
# The instances that are not sampled are called "out-of-bag" instances
# OOB evaluation: since OOB instances are not seen by a model during training,
# these can be used to estimate the performance of the ensemble without the need for CV
# Each of the N models is trained with its bootstrap sample, then evaluated
# on its OOB sample, and then you take an average of all the N OOB scores
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=6, random_state=123)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1, oob_score=True) # for regressors use R2 score
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
oob_accuracy = bc.oob_score_
print('Test set accuracy: {.3f}'.format(accuracy))
print('OOB accuracy: {.3f}'.format(oob_accuracy))

# RANDOM FOREST
# An ensemble method that uses a decision tree as a base estimator
# Each tree is trained on a different bootstrap sample with the same size as the training set
# It introduces further randomization than the bagging when training each of the base estimators
# When each tree is trained, only 'd' features (variables) can be sampled at each node without replacement
# Each node is then split using the sample feature that maximizes the information gain
# The result is that the overal variance is lower than in individual trees
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=123)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)

# FEATURE IMPORTANCE
# Measure the predictive power of a feature
# It is measured by how muc the nodes use a particular feature (weighted average) to reduce impurity
# It is expressed as a percentage indicating the weight of that feature in training and prediction
# You can get it with
model.feature_importances_
# Visualize feature importance
import pandas as pd
from matplotlib import pyplot as plt
importances_rf = pd.Series(rf.feature_importances_, index=X.columns)
sorted_importances_rf = importances_rf.sort_values()
sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.show()


# --- Ch4: BOOSTING ---
# Boosting refers to an ensemble method in which many predictors are trained and
# each predictor learns from the errors of its predecessors
# Formally, many 'weak learners' are combined to form a 'strong learner'
# A weak learner is model that does slightly better than random guessing
# For example a tree with max depth of 1 (called a 'decision stump') is a weak learner
# There are several boosting methods, we will see: ADABOOST and Gradient Boosting
# (Estimators do not necessarily have to be trees)

# ADABOOST: adaptive boosting
# Each predictor pays more attention to the instances that were wrongly predicted by its predecessor
# This is achieved by constantly changing the weights of the training instances
# Furthermore, each predictor is assigned a coefficient 'alpha' that weights
# its contribution in the ensemble's final prediction
# alpha depends on the predictor's training error
# Important parameter: the LEARNING RATE 'eta'
# It is a number between 0 and 1 that shrinks alpha
# There is a trade-off between the learning rate and the number of estimators
# A smaller value of eta should be compensated by a greater number of estimators
# For classification, the prediction is obtained by weighted mayority voting (AdaBoostClassifier)
# For regression, a weighted average of each estimator's prediction (AdaBoostRegressor)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
dt = DecisionTreeClassifier(max_depth=1, random_state=123) # a decision 'stump'. it does have to be always a stump
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
adb_clf.fit(X_train, y_train)
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# GRADIENT BOOSTING:
# As AdaBoost, each predictor corrects its predecessor's errors
# But instead of tweaking the weights of the training instances, each predictor
#is trained using the residual errors of its predecessor as labels
# GRADIENT BOOSTING TREES: gradint boosting with CART as the base learner
# Important parameter: SHRINKAGE
# The prediction of each tree in the ensemble is shrinked after it is mutiplied by 
# the learning rate 'eta' which is a number between 0 and 1
# Similar to adaboost, there is a trade-off between eta and the number of estimators
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=123)
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)

# STOCHASTIC GRADIENT BOOSTING
# Gradient boosting involves an exhaustive search procedure, as each CART is 
# trained to find the best split points and features
# This may lead to CARTS that use the same split points and possibly the same features
# To mitigate this effect you can use the Stochastic gradient boosting algorithm
# Each tree is trained on a random subset of rows of the training data
# The sampled instances (40-80% of the training set) are sampled without replacement
# At the level of each node, features are sampled without replacement when choosing the best split points
# This creates further diversity in the ensemble and the net effect is adding more variance to the ensemble of trees
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
sgbt = GradientBoostingRegressor(n_estimators=300, max_depth=1,
                                    subsample=0.8, # use 80% of the training sample
                                    max_features=0.2, # use 20% of available features in each tree
                                    random_state=123)
sgbt.fit(X_train, y_train)
y_pred = sgbt.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)

# --- Ch5: MODEL TUNING ---

# Here we will only see GridSearch

# Tuning a CART's hyperparameters
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=2, random_state=123)
# Return a dictionary with parameters as keys and default values as values
print(dt.get_params())
from sklearn.model_selection import GridSearchCV
params_dt = {
    'max_depth': [3,4,5,6],
    'min_samples_leaf': [0.04, 0.06, 0.08],
    'max_features': [0.2, 0.4, 0.6, 0.8]
}
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt,
                        scoring='accuracy', cv=10, n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters: /n', best_hyperparams)
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))
best_model = grid_dt.best_estimator_ # this model is fitted on the whole training set because the 'refit' parameter of GridSearchCV is True by default
test_acc = best_model.score(X_test, y_test)

# Tuning a RANDOM FOREST's hyperparameters
# In addition to all the hyperparameters of the CARTs in the random forest,
# there are other hyperparams like the number of estimators, whether it uses boostrapping or not, etc.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=123)
rf.get_params()
params_rf = {
    'n_estimators': [300,400,500],
    'max_depth': [4,6,8],
    'min_samples_leaf': [0.1, 0.2],
    'max_features': ['log2', 'sqrt'] 
}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, verbose=1,
# 'verbose' controls verbosity, the higher its value the more messages are printed during fitting
                        scoring='neg_mean_squared_error', cv=3 n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_hyperparams = grid_rf.best_params_
print('Best hyperparameters: /n', best_hyperparams)
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)


#%% EXTREME GRADIENT BOOSTING WITH XGBOOST

# XGBOOST is a library for optimized gradient-boosting ML models
# Applies to supervised learning problems
# Benefit: speed (parallelizable) and performance

# It is an ENSEMBLE LEARNING method

# BOOSTING: an ensemble meta algorithm used to convert many weak learners into
# a strong learned. It is implemented by iteratively learning a set of weak 
# models on subsets of the data, weighing each weak prediction according to each
# weak learner's performance, and then combining all the weighted predictions to 
# obtain a single weighted prediction

# WEAK LEARNER: a model that barely outperfoms random guessing

# When it SHOULD be used: supervised ML task that fits the following criteria:
# - Large N of training samples (>1000 and <100 features)
# - N features < N training samples
# - When you have a mixture of categorical and numeric features, or just numerical

# When it SHOULD NOT be used (suboptimal compared to deep learning):
# - Image recognition
# - Computer vision
# - NLP

# One data set used here:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques


# --- Ch1: Classification with XGBoost ---

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X, y = dt.iloc[:,:-1], df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initiate XGBoost classifier instance
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# CROSS VALIDATION is implemented differently with this API, it is already built in
import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv('classification_data.csv')

# Convert data set into optimized structure (DMatrix)
# (in the previous example, with XGBClassifier the data was converted on the fly.
# But when using the xgb object you have to do this beforehand)
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:,:-1], # X variables
                            label=chrun_data.month_5_still_here # Y variable
                            )

# Set the parameters for each tree
params = {"objective":"binary:logistic", "max_depth":4}

# Implement CB with the .cv method
cv_results = xgb.cv(dtrain=churn_dmatrix, 
                    params=params, 
                    nfold=4, # n of folds
                    num_boost_round=10, # number of trees
                    metrics='error', # metric to compute (you could use "auc" for example)
                    as_pandas=True)
print("Accuracy: %f" %((1-cv_results["test-error-mean").iloc[-1]]))


# --- Ch2: Regression with XGBoost ---

# The metrics will now be RMSE or MAE (mean absolute error)

# LOSS FUNCTION: quantifies how far off a prediction is from the actual result.
# It measures the difference between estimated and true values for some collection of data
# The goal is to find the model that minimizes the loss function.

# COMMMON LOSS FUNCTIONS FOR XGBOOST:
# For regression: 'reg:linear'
# For binary classification problems, when you care about the decision and not the probability: 'reg:logistic'
# For binary classification problems, when you care about the probability: 'binary:logistic'

# The goal of XGBoost is to have base learners that are slightly better than random guessing on certain subsets of 
# training examples, and uniformly bad at the remainder. So that, when all the predictions are combined, the uniformly 
# bad predictions cancel out, and those slightly better learners combine into a single very good non-linear prediction.

# Comparing BASE LEARNERS in XGBoost
# LINEAR BASE LEARNER:
# - A sum of linear terms, exactly as you would find in a logistic or linear regression model-
# - When you combine them into an ensemble, you get a weighted sum of linear models, which is itself linear. Therefore,
# you do not get any non-linear features in the final model. Very similar results can be obtained with any other linear
# model, that's why this method is rarely used.
# TREE BASE LEARNER:
# - A decision tree
# - When they are combined into an ensemble, the combination becomes a non-linear function of each individual tree, 
# which is non-linear

# XBGOOST with TREES as base learners using XGBoost Scikit-Learn-compatible API
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
X, y = boston_data.iloc[:, :-1], boston_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)

preds=xg_reg.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# Repeat but without using the Scikit-Learn API, just use the Learning API in XGBoost
# (and use a linear learner as the base learner)
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
X, y = boston_data.iloc[:, :-1], boston_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)

params = {"booster":"gblinear", # base learer (linear learner insteaed of tree)
          "objective":"reg:linear"}

xg_reg = xgb.train(params=params, dtrain=DM_train, 
                   num_boost_round=10) # boosting rounds

preds=xg_reg.predict(DM_test)
rmse=np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# CROSS VALIDATION
housing_dmatrix = xgb.DMatrix(data=X,label=y)

params = {"objective":"reg:linear", # loss function
          "max_depth":4}

cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

print(cv_results)
print((cv_results["test-rmse-mean"]).tail(1))


# REGULARIZATION IN XGBOOST
# XGBoost does regularization.
# Parameters for regularization:
# For tree-based learners:
# - gamma: minimum loss reduction allowed for a split to occur. It controls whether a given 
# node on a base learner will split based on the expected reduction in the loss that would occur after performing the 
# split. Higher values lead to fewer splits.
# - alpha: l1 regularization on leaf weights. Higher values lead to stronger regularization
#   which causes many leaf weights in the base learners to go to zero.
# - lambda: l2 regularization on leaf weights. Smoother than l1. Causes leaf weights to smoothly decrease.

# l1 REGULARIZATION with XGBoost
import xgboost as xgb
import pandas as pd

boston_data = pd.read_csv('boston_housing.csv')
X, y = boston_data.iloc[:, :-1], boston_data.iloc[:, -1]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

params = {"objective":"reg:linear", "max_depth":4}

l1_params = [1, 10, 100] # we will try these l1 values
rmses_l1 =[]
for reg in l1_params:
    params['alpha'] = reg
    cv_results = xgb.cv(dtrain=boston_dmatrix, params=params, nfold=4, num_boost_round=10,
                                                     metrics='rmse', as_pandas=True, seed=123)
    rmses_l1.append(cv_results['test-rmse-mean'].tail(1).values[0])
print('Best rmse as a function of l1:')
print(pd.DataFrame(list(zip(l1_params, rmses_l1))), columns=['le', 'rmse'])


# l2 REGULARIZATION with XGBoost
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

params = {"objective":"reg:linear", "max_depth":3}
rmses_l2 = []
for reg in reg_params:
    params["lambda"] = reg
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    rmses_l2.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])

print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


# VISUALIZING individual XGBoost trees
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":2}
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)
# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()
# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()
# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()


# VISUALIZING the most import features in the dataset for the model
# One simple way of doing this involves counting the number of times each feature is split on across all boosting rounds
#  (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many
#  times they appear.
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear", "max_depth":2}
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)
# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()


# --- Ch3: Fine-tuning your XGBoost model ---

# Example: tune number of roundsimport xgboost as xgb
import pandas as pd
import numpy as np

housing_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"reg:linear", "max_depth":3}

num_rounds = [5, 10, 15]
final_rmse_per_round = []
for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))

# Let XGBoost automatically select th best value for the parameter using EARLY STOPPING
# Early stopping works by testing the XGBoost model after every boosting round against a hold-out dataset and stopping 
# the creation of additional boosting rounds (thereby finishing training of the model early) if the hold-out metric 
# ("rmse" in our case) does not improve for a given number of rounds.
# Bear in mind that if the holdout metric continuously improves up through when num_boost_rounds is reached, then early 
# stopping does not occur
housing_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:linear", "max_depth":4}

cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, 
                    num_boost_round=50, 
                    early_stopping_rounds=10,
                    metrics="rmse", as_pandas=True, seed=123)
print(cv_results)

# Overview of XGBoost's hyperparameters
# The parameters that can be tuned depend on the base learner you use

# For TREES:
# - learning rate ('eta'): affects how quicky the model fits the residual error using additional base learners. A model with a 
# low learning rate will require more boosting rounds to achieve the same reduction in residual error
# - gamma: min loss reduction to create new tree split
# - lambda: L2 reg on leaf weights
# - alpha: L1 reg on leaf weights
# - max_depth: how deeply each tree is allowed to grow doing any boosting round
# - subsample: the fraction of the total training set that can be used for any boosting round. A low value may lead to
# underfitting, and a high value to over-fitting
# - colsample_bytree: the fraction of features you can select from during any bootsing round. A large value means that
# almost all features can be ued to build a tree during a given boosting round. In genearal, smaller values can be 
# thought of providing addition regularization, while high values can lead to over-fitting. (Equivalent to 
# "max_features" in Random Forest)

# For LINEAR BASE LEARNERS:
# - lambda: L2 reg on weights
# - apha: L1 reg on weighst
# - lambda_bias: L2 reg term on bias

# The number of boosting rounds, that is, the number of trees or base learners you construct, is also a tuneable parameter


# Example: tune the LEARNING RATE
housing_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"reg:linear", "max_depth":3}

eta_vals = [0.001, 0.01, 0.1]
best_rmse = []
for curr_val in eta_vals:
    params["eta"] = curr_val    
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, 
                    num_boost_round=10, 
                    early_stopping_rounds=5,
                    metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))


# Example: tune MAX_DEPTH
housing_dmatrix = xgb.DMatrix(data=X,label=y)

params = {"objective":"reg:linear"}

max_depths = [2,5,10,20]
best_rmse = []

for curr_val in max_depths:
    params["max_depth"] = curr_val
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, 
                    num_boost_round=10, 
                    early_stopping_rounds=5,
                    metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))


# Example: tune COLSAMPLE_BYTREE
housing_dmatrix = xgb.DMatrix(data=X,label=y)

params={"objective":"reg:linear","max_depth":3}

colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:
    params['colsample_bytree'] = curr_val
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))


# Review of GRID SEARCH and RANDOM SEARCH

# GRID SEARCH: exhaustibly searching through a collection of values
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
housing_data = pd.read_csv('hdata.csv')
X, y = housing_data[housing_data.columns.to_list()[:-1]], housing_data[housing_data.columns.to_list()[-1]]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Define parameter grid
gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 0.9],
                  'n_estimators': [200],
                  'subsample':[0.3, 0.6, 0.9]}
                  # 12 combinations
gmb = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', # metric
                        cv=4, # cv folds
                        verbose=1) # define kind of output
grid_mse.fit(X, y)
print('Best parameters found: ', grid_mse.best_params_)
print('Lowest RMSE found: ', np.sqrt(np.abs(grid_mse.best_score_)))

# Disadvantages: as the nº of values increases, the combinatios and therefore the time increase exponentially


# RANDOM SEARCH: you create a range of hyperparameter values you would like to search over, and
#   set the number of iterations you would like for the random search to continue. You draw a random set of values a 
#   defined amount of times, and then pick the best one.
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
housing_data = pd.read_csv('hdata.csv')
X, y = housing_data[housing_data.columns.to_list()[:-1]], housing_data[housing_data.columns.to_list()[-1]]
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Define parameter grid
gbm_param_grid = {'learning_rate': np.arrange(0.05, 1.05, .05),
                  'n_estimators': [200],
                  'subsample': np.arrange(0.05, 1.05, .05)} 
                  # if you ran a grid search, these would amount to 400 combinations
gmb = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, 
                              param_distributions=gbm_param_grid,
                              n_iter=25, # number of iterations of the random search (n of combinatios)
                              scoring='neg_mean_squared_error', # metric
                              cv=4, # cv folds
                              verbose=1)
randomized_mse.fit(X, y)
print('Best parameters found: ', randomized_mse.best_params_)
print('Lowest RMSE found: ', np.sqrt(np.abs(randomized_mse.best_score_)))

# Disadvantage: as you add new values the parameter space becomes exponentially large and it reduces the chances of
#   getting the best combination, so it takes more time.


# --- Ch4: Using XGBoost in pipelines ---

# PREPROCESSING strategy: 
# 1) Use LabelEncoder and OneHot Encoder (which cannot be run in a Pipeline)
# LabelEncoder: converts a categorical column of strings into integers that map onto those strings
# OneHotEncoder: takes the column of integers and encodes them as dummy variables

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df =pd.read_csv('ames_housing_data.csv')  https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import LabelEncoder
df.LotFrontage = df.LotFrontage.fillna(0)
categorical_mask = (df.dtypes == 'object')
categorical_columns = df.columns[categorical_mask].tolist()

# Create LabelEncoder object: le
le = LabelEncoder

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])


# 2) DictVectorizer: converts lists of feature mapping into vectors
# You first need to convert the DataFrame into a list of dictionary entries
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)


# Incorporating XGBoost into pipelines
# Use XGBoost scikit learn API within a pipeline object
import pandas as pdimport xgboost as xgb
import numpy as npfrom sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age',
         'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']
data = pd.read_csv("boston_housing.csv", names=names)
X, y = data.iloc[:,:-1], data.iloc[:,-1]
xgb_pipeline = Pipeline([('st_scaler', StandardScaler()),
                        ('xgb_model', xgb.XGBRegressor())])
# Compute the corss validated negative mse using 10-fold CV
scores = cross_val_score(xgb.pipeline, X, y, scoring="neg_mean_squared_error", 
                         cv=10)
# Convert the 10-fold negative MSE into an average RMSE across all 10 folds
final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print('Final XGB RMSE:", final_avg_rmse')


# END-TO-END PIPELINE WITH XGBOOST EXAMPLE

# We will have to use the library sklearn_pandas to "bridge the gap" between pandas and sklearn
# It has a generic class called DataFrameMapper which allows for easy conversion between scikit learn
# aware objects, or pure numpy arrays, and DataFrames

# We will also use uncommon aspects of sklearn:
# sklearn.impute: SimpleImputer class to fill in missing numerical and cagtegorial values
# sklearn.pipeline: FeatureUnion class allows to combine separate pipeline outputs into
#   a single output. For example, we would need to do this if we had one set of pre-processing
#   steps on the categorical part of the dataset, and another one on the numerical part of the dataset

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

         # Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict('records'),
y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))ç

# Kidney disease case study I: Categorical Imputer

# You'll now continue your exploration of using pipelines with a dataset that requires significantly more wrangling. 
# The chronic kidney disease dataset contains both categorical and numeric features, but contains lots of missing values.
# The goal here is to predict who has chronic kidney disease given various blood indicators as features.

# As Sergey mentioned in the video, you'll be introduced to a new library, sklearn_pandas, that allows you to chain many
#  more processing steps inside of a pipeline than are currently supported in scikit-learn. Specifically, you'll be able
#  to use the DataFrameMapper() class to apply any arbitrary sklearn-compatible transformer on DataFrame columns, where
#  the resulting output can be either a NumPy array or DataFrame.

# We've also created a transformer called a Dictifier that encapsulates converting a DataFrame using .to_dict("records")
#  without you having to do it explicitly (and so that it works in a pipeline). Finally, we've also provided the list of
#  feature names in kidney_feature_names, the target name in kidney_target_name, the features in X, and the target in y.

# In this exercise, your task is to apply sklearn's SimpleImputer to impute all of the categorical columns in the dataset.
#  You can refer to how the numeric imputation mapper was created as a template. Notice the keyword arguments 
# input_df=True and df_out=True? This is so that you can work with DataFrames instead of arrays. By default, the 
# transformers are passed a numpy array of the selected columns as input, and as a result, the output of the DataFrame
#  mapper is also an array. Scikit-learn transformers have historically been designed to work with numpy arrays, 
# not pandas DataFrames, even though their basic indexing interfaces are similar.

from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy="median")) for 
    numeric_feature in non_categorical_columns],
    input_df=True,
    df_out=True
    )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [(category_feature, SimpleImputer(strategy="median")) for 
    category_feature in categorical_columns],
    input_df=True,
    df_out=True
    )

# Having separately imputed numeric as well as categorical columns, your task is now to use scikit-learn's FeatureUnion 
# to concatenate their results, which are contained in two separate transformer objects - numeric_imputation_mapper, 
# and categorical_imputation_mapper, respectively.
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
    ("num_mapper", numeric_imputation_mapper),
    ("cat_mapper", categorical_imputation_mapper)
])

# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, 
kidney_data, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))


# HYPERPARAMETER TUNING
# In order for the hyperparameters to be passed to the appropriate step, you have to name the parameters in the dictionary
# with the name of the step being referenced, followed by two underscore signs and the name of the hyperparameter you
# want to iterate over
# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline, 
param_distributions=gbm_param_grid, scoring='roc_auc', n_iter=2, cv=2,
verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print('Best estimator found: ', randomized_roc_auc.best_estimator_)
print('Lowest RMSE found: ', randomized_roc_auc.best_score_)


# WHAT THE COURSE LEFT OUT:
# Using XGBoost for ranking/recommendation probmes (modify the loss function)
# More powerful hyperparameter tuning strategies (like Bayesian Optimization)
# Usign XGBoost as part of an ensemble of other models


#%% PREPROCESSING FOR MACHINE LEARNING IN PYTHON

# Ch1: INTRODUCTION TO DATA PREPROCESSING

# Recap: data exploration with pandas
import pandas as pd
print(df.head())
df.info() # number of columns and rows, columns, data type, non-missing values
df.describe() # summary statistics
df.var() # check variance of columns
df.isna().sum() # count missing values
df.dropna() # dro pall rows containing missing values
df.drop("Col1", axis=1) # drop an entire column
df.dropna(subset=["Col1"]) # keep all non-missing values according to a column
df.dropna(thresh=2) # drop all rows with less than 2 non-missing values

# Working with data types
df['col'] = df['col'].astype("float") # convert data type to float
print(df.dtypes)

# Stratified sampling into train and test set
# Make sure that the class distribution in the train and test sets are the same
#   as the distribution of the original sample
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_State=123)


# Ch2: STANDARDIZING DATA

# Transform continuous data to appear normally distributed. This is often necessary
# because models assume normally distributed data.

# Scenarios where you would want to standardize data:
# - Working with any kind of model that uses a linear distance metric, or operates
#   in a linear space (kNN, linear regression, K-means Clustering)
# - When features have a high variance, which could bias a model tha assumes data is normally distributed.
#   If a feature in the data has a very high variance, this could impact the model's ability to learn from the other features.
# - A lot of continuous features with different scales (e.g.: house price prediction).

# However, there are models that have no problem working in a non-linear space, or do standardization in estimation.

# LOG-NORMALIZATION: useful when you have features with high variance.
# Applies logarithmic transformation to turn data into a scale that approximates normality
import numpy as np
df['col1'] = np.log(df['col1'])

# SCALING DATA FOR FEATURE COMPARISON
# Most useful when working with data with continuous features with different scales and
# with models that operate in some sort of linear space (linear regression, kNN)
# I transforms the features so that they have mean 0 and variance 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), colums=df.columns)
# Apply this into a workflow. E.g. kNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()
scaler = StandardScaler()
# !!! It is very important to scale the data AFTER splitting it into train and test
# Otherwise you are doing data-leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # It is important to use the transform method
# instead of fit_transform here so that the test features won't be used to fit the model.
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)


# Ch3: FEATURE ENGINEERING

# Extract and expand information from existing features to improve performance ad gain insights into relationships between features
# It is very specific to each dataset

# ENCODING CATEGORICAL VARIABLES

# ENCODING A BINARY VARIABLE
# with pandas
df['col_enc'] = df['col'].apply(lambda val: 1 of val=='y' else 0)
# with Sklearn
# This is useful when the encoding is part of a scikit-learn pipeline. Creating a LabelEncoder
# also allows us to use the same encoding in another data.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col_enc'] = le.fit_transform(df['col'])

# ONE-HOT ENCODING: encodes categorical variables into 1s and 0s when there are more than
# two values to encode. It transforms unique values of the column into an array, and
# shows a 1 in the place of each unique value when a particular record has that value
# You can turn this into dummy variables with pandas
pd.get_dummies(df['cat_col'])

# ENGINEERING NUMERICAL FEATURES
# Aggregate statistics and dates
# Average across columns
df['mean_col'] = df.loc[:, 'day1':'day5'].mean(axis=1)
# Reduce the granularity of a date variable
df['date_converted'] = pd-to_datetime(df['date'])
df['month'] = df['date_converted'].dt.month

# ENGINEERING TEXT FEATURES
# 1) EXTRACTION WITH REGEX
import re
my_string = 'temperature: 75.6 F'
temp = re.search('\d+\.\d+', my_string)
# \d is digits, + is to grab as many as possible, \. escape the decimal point

# 2) VECTORIZING TEXT: transform text into a numerical input.
# TF/IDF (term frequency/inverse document frequency): reflects how important a word
# is in a document beyond how it frequently it occurs. Places the weight on words that are ultimately more significant
# in the entire corpus of words.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(documents)
# Now that we have a vectorized version of the text, we can use it for classification using a Naive Bayes, for example


# Ch4: SELECTING FEATURES MODELING

# Remove correlated features by calculating Pearson's correlation coefficient
print(df.corr())

# SELECTING FEATURES USING TEXT VECTORS
# If you have a tfid vector, you do not necessarily need to use it all. You
#   could use a % of the words by looking the weights of the words.
# After you vectorize the text, the vocabulary and weights will be stored in the
#   vectorizer.
# See vocabulary weights
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(documents)
print(tfidf_vec.vocabulary_)
# Access the weights of a row
print(text_tfidf[3].data)
# Access the indices of words that have been weighted
print(text_tfidf[3].indices)
# Reverse the key-balue pairs in the vocabulary dictionary
vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()}
# Zip row indices and weights
zipped_row = dict(zip(text_tfidf[3].indices,
                      text_tfidf[3].data))
# Or do it in a function (create mapping of words to weights)
def return_weights(vocab, vector, vector_index):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    return {vocab[i]:zipped[i] for i in vector[vector_index].indices}
print(return_weights(vocab, text_tfidf, 3))

# Ch5: PUTTING IT ALL TOGETHER


#%% UNDERSTANDING DATA ENGINEERING

# It is a conceptual course, there is no coding involved.

# Ch1: WHAT IS DATA ENGINEERING?

# 4 general steps through which data flows in an organization:
# 1) Data collection and storage, 2) Data preparation
# 3) Exploration and visualization, 4) Experimentation and prediction
# Data engineers are responsible for the first step of the process
# Their job is to deliver the data in the right form, to the right people, as efficiently as possible
# They ingest data from different sources, optimize dbs for analyses, remove corrupted data,
# develop,construct, test and maintain data architecures
# They build data pipelines

# DATA ENGINEERS VS DATA SCIENTISTS: 
# Data scientists intervine on the other 3 parts of the workflow
# Data engineers enable data scientists

# THE DATA PIPELINE: companies ingest data from many different sources
# which need to be processed and stored in various ways.
# To do that we need DATA PIPELINES which efficiently automate the flow from
# one station the next so that data scientists can use it
# They involve automating the extracting, transforming, combining, validating and loading of data
# They reduce human intervention, errors, and the time it takes for data to flow

# ETL PIPELINES: one popular framework for designing data pipelines
# It breaks up the flow of data into 3 sequential steps:
# 1) Extract data, 2) Transforming ir, 3) Loading transformed data to another database
# But not all pipelines are ETL (eg. the data may not be trasnformed)


# Ch2: STORING DATA

# DATA STRUCTURES

# STRUCTURED DATA: easy to search and organize, follows a consistent model
# (eg. rows and cols), has defined data types, can be easily grouped to form relations,
# it is stored in a relational database, created and queried using SQL

# SEMI-STRUCTURED DATA: relatively easy to organize, less rigid implementation,
# diffrent types, can be grouped but needs more work, stored in NoSQL dbs like JSON, XML, YAML file formats)
# Different observations may have different sizes

# UNSTRUCTURED DATA: it does not follow a model, can't be contained in row and column format, 
# it is difficult to search and organize, examples are text, sound, pictures, videos,
# usually stored in data lakes, data warehouses or dbs.
# In the case of Spotify, it could be lyrics, songs, album's pictures.
# It can be processed by AI and machine learning.
# Adding information to categorize it would make it semi-structured (like adding a genre to each song)

# SQL DATABASES: the industry standard for querying Relational DB Management System (RDBMS)
# 2 main advantages: allows you to access many records at once, and group, filter and aggregate them
# Close to English, so easy to write and understand
# Engineers use it to create and maintain databases
# D Scientists use it to query dbs

# DATABASE: organized data, stored and accessed on a computer
# Relational dbs and data warehouses are examples of dbs

# DATA LAKE: stores all raw data (unprocessed and messy),
# stores big data, stores all data structures so it does not enforce any
# model on the way to store the data, which makes it cost-effective, but it
# is very difficult to analyse it, requires an up-to-date catalog
# used by data scientists for real-time analytics on big data

# DATA WAREHOUSE: stores specific data for a specific use,
# usually relatively small (to big data), stores mainly structred data
# so it is more costly to update, but it is optimized for analytics,
# they are used by analyst for ad-hod read-only queries.

# DATA CATALOG: "a source of truth that compensates for the lack of structure in a data lake"
# It stores info on the data sources, how the data is used, who is responsible for
# maintainin it, and how often it is updated. Good practice and ensures reproducibility.


# Ch3: MOVING AND PROCESSING DATA

# To process data is to convert raw data into meaningful information
# We may need to remove unwanted data, optimze memory/process/network costs,
# convert data from one type to another, organize data, fit the data into
# a schema/structure, increase productivity

# SCHEDULING DATA: this can apply to any data processing task, but we will focus
# on updating tables and dbs. It is the "glue" of a data engineering system
# as it holds each small piece and organize how they work together by
# running tasks in a specific order and resolving all dependencies

# There are differnt ways to do this: we can run tasks manually,
# autmatically at specific times, automatically if a specific condition
# is met (SENSOR SCHEDULING, requires "sensors" always "listening"
# if something is being changed)

# HOW THE DATA IS INGESTED: it can be ingested in BATCHES (sent by groups
# at specific intervals, cheaper) or STREAMS (individual data records are sent
# through the pipeline as soon as they are updated, eg registering a new user)

# PARALLEL COMPUTING/PROCESSING: the basis of almost all moder data processing tools
# When big data processing tools perform a processing task, they split it into smaller tasks
# which are sent to different computers
# Advantage: extra processing power, reduced memory footprint (as you can partition the data)
# Disadvantages: costs, re-arranging split data requires communication time between computers
# (so you have to consider if the time gained by using multiple processing units
# is greater than the time lost communicating them)

# CLOUD COMPUTING
# Companies can process data on their own data centers, often on premise
# Or they can rent servers in the cloud


#%% INTRODUCTION TO DATA ENGINEERING

# Ch1: INTRO

# An engineer that develops, constructs, tests, and maintains architectures such
# as datbases and large-scale processing systems

# The job of a data engineer is making the life of the data scientist easier, by
# gathering data, optimize dbs for analyses, and remove corrupt data

# TOOLS of a Data Engineer:
# - Database systems (eg. MySQL, PostgreSQL, MongoDB)
# - Processing tools (parallel processing, Apache Spark, Hive)
# - Scheduling tools (plan jobs with specific intervals, 
# resolve dependency requirements of jobs eg. the order of operations. Apache Airflow, Oozie, Linux Cron)

# CLOUD COMPUTING
# Cloud service providers offer tools for storage, processing, and databases


# Ch2: DATA ENGINEERING TOOLBOX

# DATABASES: holds data, organizes data, retrieves data through a DBMS

# STRUCTURED DATA: choherent to a well-defined structure (sometines defined by a schema),
# SEMI-STRUCTURED DATA: eg. jsons
# UNSTRUCTURED DATA: have no schema

# SQL DB: has tables, a schema, is relational
# NoSQL DB: non-relational, structured or unstructured (it is a misconception to relate it to unstructured,
# schema-less data). Eg: Redis (key-value stores), MongoDB (documents)

# DATABASE SCHEMA: describes a structure and relations of a db
# SATAR SCHEMA: one or more fact tables referencing any number of dimension tables (a type of schema).
# Fact tables represent things that happened (eg orders), and dimension tables represent information (eg customer data)

# Query an SQL database with Pandas (PostgreSQL)
import pandas as pd
data = pd.read_sql("""
SELECT first_name, last_name FROM "Customer"
ORDER BY last_name, first_name
""", db_engine)
data = pd.read_sql("""
SELECT * FROM "Customer"
INNER JOIN "Order"
ON "Order"."customer_id"="Customer"."id"
""", db_engine)

# PARALLEL COMPUTING: split a task into subtasks and distribute them over several computers
# Benefits: extra processing power, less memory footprint
# Costs: splitting and re-merging requires communication time, so tasks need to be large.
    # "Parallel slowdown": due to the communication overhead, speed does not increase linearly

# PARALLEL PROCESSING IN PYTHON - Using multiple cores of the same machine
# Eg: calculate the average age of participants for each Olympic
from mutiprocessing import Pool
def take_mean_age(year_and_group):
    year, group = year_and_group
    return pd.DataFrame({"Age": group["Age"].mean()}, index=[year])
# The func returns a df with 1 obs and 1 col with the mean age of the group, indexed by year
# Here use 4 cores
with Pool(4) as p:
    results = p.map(take_mean_age, athlete_events.groupby("Year"))
    results_df = pd.concat(results)

# PARALLEL PROCESSING IN PYTHON - Using the Dask package
import dask.dataframe as dd
# Partition the df into 4:
athlete_events_dask = dd.from_pandas(athlete_events, npartitions=4)
# Run parallel computations on each partition
results_df = athlete_events_dask.groupby("Year").Age.mean().compute()

# Function to apply a function over multiple cores
@print_timing # time each operation
def parallel_apply(apply_func, groups, nb_cores):
    with Pool(nb_cores) as p:
        results = p.map(apply_func, groups)
    return pd.concat(results)
# Parallel apply using 4 cores
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 3)

# PARALLEL COMPUTING FRAMEWORKS

# APACHE HADOOP: a collection of open source proyects, including MAPREDUCE and HDFS

# HDFS: distributed file system (nowadays cloud management storage systems ofter replace HDFS)

# MAPREDUCE: one of the first big data processing paradigm, it splits tasks into subtasks over a cluster

# HIVE: a layer on top of the Hadoop ecosystem that makes data from several sources queriable in 
# a structured way using Hive SQL. Initially created by Facebook.
# a HIVE QUERY looks like an SQL query, but behind the curtains it transforms it into a job that 
# can operate on a cluster of computers via Hadoop's Mapreduce

# APACHE SPARK: distributes data processing tasks between clusters of computers.
# Unlike Mapreduce, it tries to keep as much processing as possible in memory, avoiding disk writes
# Its architecture relies on Resilient distributed datasets (RDDs), a data strucutre that maintains data
# which is distributed into multiple nodes, RDDs dont have named columns, you can think of them
# lists of tuples. You can do 2 types of operations: transformations like .map() or .filter()
# and actions like .count() or .first()
# PYSPARK: the Python interface to Spark, similar to pandas, instead of using the SQL format

# Use PYSPARK
print(type(athlete_events_spark)) # <class 'pyspark.sql.dataframe.DataFrame'>
# Print the schema of an SQL database with PySpark
print(athlete_events_spark.printSchema())
# Group by the Year, and find the mean Age
print(athlete_events_spark.groupBy('Year').mean('Age').show()) # display a PySpark db with .show()

# Usually you create a .py Spark file and run it in the console like this
spark-submit master local[4] /folder/sparkfile.py

# WORKFLOW SCHEDULING FRAMEWORKS
# Example of a pipeline: extract data from csvs, clean it with spark, and load it to an SQL db
# You could run this periodically by hand, or schedule it
# Linux's CRON is a useful tool to schedule simple tasks individually
# but when there are several tasks with dependencies, it is not enough
# DAG: directed acyclic graph. A set of nodes connected by directed edges
# THere are no cycles in the graph, meaning that no path following the directed edges
# sees a specific node more than once

# SPOTIFY's LUIGI: allows for the definition of DAGs for complex pipelines

# APACHE AIRFLOW: Python workflow scheduler
# created by Airbnb, then open-sourced. Built around the concept of DAGs

# Example: a DAG that starts by starting a cluster, then it ingests cutomer data
# and ingests product data, and once those two are finished, it enriches customer data
# Create the DAG object
dag = DAG(dag_id='example_dag', ..., schedule_interval="0 * * * *")
# this is crontab notation.
# The leftmost character represents minutes, then hour, day of the month, month, day of the week
# Run every hour at minute N would be "N * * * *"
# Define operations
start_cluster = StartClusterOperator(task_id="start_cluster", dag=dag)
ingest_customer_data = SparkJobOperator(task_id="ingest_customer_data", dag=dag)
ingest_product_data = SparkJobOperator(task_id="ingest_product_data", dag=dag)
enrich_customer_data = PythonOperator(task_id="enrich_customer_data", dag=dag)
# Set up dependency flow
start_cluster.set_downstream(ingest_customer_data)
ingest_customer_data.set_downstream(enrich_customer_data)
ingest_product_data.set_downstream(enrich_customer_data)


# Ch3: EXTRACT, TRANSFORM AND LOAD (ETL)

# EXTRACTING DATA: taking data from persistent storage (like a file on Amazon S3 or a SQL db)
#  which is not suited for data processing into memory
# - FROM TEXT FILES: unstructured (like a book chapter) or flat files (rows and cols, csv),
 # semi/scturctured like JSON: 4 atomic data types (number, string boolean and null) and 2
 # composite data types (arrays and objects), similar to Python dictionaries

# Parse JSON data
import json
result = json.loads('{key_1': 'value_1', 'key_2': 'value_2')
print(result['key_1'])

# Make a get request to an API
import requests
response = requests.get("https://hacker-news.firebaseio.com/v0/item/1516800333.json")
print(response.json()) # .json() parses the income json and transforms it to a Python object

# EXTRACT FROM DATABASES
# Two main db types:
# APLICATION DATABASES: optimized for transactions, each transcation changes or inserts
# rows in a db, called OLTP ("Online transaction processing"), row-oriented
# ANALYTICAL DATABASES: optimized for analysis, OLAP ("On-line Analytical processing"), column-oriented

# To extract data from a db in Python you will need a CONNECTION STRING/URI (a 
# string that holds info on how to connect to a db)
# Example: postgresql://[user[:password]@][host][:port]
# CONNECT TO AN SQL DATABASE
import sqlalchemy
connection_uri = "postgresql://repl:password@localhost:5432/paglia"
db_engine = sqlalchemy.create_engine(connection_uri)
import pandas as pd
pd.read_sql("SELECT * FROM customer", db_engine)

# Function to extract table to a pandas DataFrame
def extract_table_to_pandas(tablename, db_engine):
    query = "SELECT * FROM {}".format(tablename)
    return pd.read_sql(query, db_engine)

# TRANSFORM
# Examples of common transformations: select an attribute, translate code values,
# validate data, splitting columns into multiple columns, joining from multiple sources

# TRANSFOMRING IN PYSPARK
# First extract the data
import pyspark.sql
spark = pyspark.sql.SparkSession.builder.getOrCreate() # spark session object
spark.read.jdbc("jdbc:postgresql://localhost:5432/paglia", # jdbc helps spark connect to rdbs
    "customer",
    properties = {"user": "repl", "password":"password"})
# Perform a join between two tables
customer_df # PySpark DataFrame with customer data 
ratings_df # PySpark DataFrame with ratings data
# Group by ratings
ratings_per_customer = ratings_df.groupBy("customer_id").mean("rating")
# Join on customer ID
customer_df.join(
    ratings_per_customer,
    customer_df.customer_id = ratings_per_customer.customer_id
)


# LOADING

# DBs for Analytics (a) vs DBs for Applications (b)
# Complex aggregate queries run on (a). Also OLAP
# COLUMN-ORIENTED, stores data per entire columns, better for parallelization
# Many transactions per second run on (b). Also OLTP
# Most (a) are ROW-ORIENTED, we store data per record

# MPP DBs: massively parallel processing databases
# Examples: Amazon Redshift, Azure SQL Data Warehouse, Google BigQuery

# Load data into Amazon Redshift
# Save a dataframe as a patquet file
    # With pandas
df.to_parquet("./s3://path/to/bucket/customer.parquet")
    # With PySpark
df.write.parquet("./s3://path/to/bucket/customer.parquet")
# Connect to Redshift and copy the data
COPY customer
FROM './s3://path/to/bucket/customer.parquet'
FORMAT AS parquet
...

# Load to a PostgreSQL database with Pandas
    df.to_sql("df", db_engine, schema="store", if_exists="replace")

# PUTTING IT ALL TOGETHER
# Ideally, the ETL should be encapsulated in an ETL function
# Function to extract PostgreSQL table to a Pandas dataframe
def extrat_table_to_df(tablename, db_engine):
    return pd.read_sql("SELECT * FROM {}".format(tablename), db_engine)
# Generic transformation function
def split_columns_transform(df, column, pat, suffixes)>
    # Converts columns into strings and splits it on pat...
# Function to load the transformed data into the PostgreSQL database
def load_df_into_dwh(film_df, tablename, schema, db_engine):
    return df.to_sql(tablename, db_engine, schema=schema, if_exists="replace")
# Open the SQL engine
db_engines = { ... } # Needs to be configured
# Define the ETL FUNCTION
def etl():
    # Extract
    film_df = extrat_table_to_df("film", db_engines['store'])
    # Transform
    film_df = split_columns_transform(film_df, "rental_rate", ".", ["_dollar", "_cents"])
    # Load
    load_df_into_dwh(film_df, "film", "store", db_engines['replace'])
# Now schedule the workflow using Apache Airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
dag = DAG(dag_id='sample', ..., schedule_interval="0 0 * * *")
etl_task = PythonOperator(task_id="etl_task", python_callable=etl, dag=dag)
etl_task.set_upstream(wait_for_this_task) # etl task will run only after wait_for_this_task is complete
# Once you have this DAG definition, you can write it into a python file and place it in the DAG folder of Airflow


# Ch4: CASE STUDY: DATACAMP
# We will use DataCamp's course ratings data.
# The objective is to start from the data about course ratings from the DataCamp
# application, which is stored in a PostgreSQL db, then clean the data and 
# build a recommendation algorithm, and then load it to the SQL Data Warehouse
# so it is available for use in a product
# We will use a table called courses and a table called rating
# Transformation function
def transform_avg_rating(rating_data):
    # Group by course_id and extract average rating per course
    avg_rating = rating_data.groupby('course_id').rating.mean()
    # Return sorted average ratings per course
    sort_rating = avg_rating.sort_values(ascending=False).reset_index()
    return sort_rating
# Extract the rating data into a DataFrame    
rating_data = extract_rating_data(db_engines)
# Use transform_avg_rating on the extracted data and print results
avg_rating_data = transform_avg_rating(rating_data)
print(avg_rating_data)
# The recommendation strategy will be: use the language that the user rated the 
# most, don't recommend courses that have already been rated by that user, then
# recommend the three highest rated courses from the remaining combinations
# Connect to the db
course_data = extract_course_data(db_engines)
# Check missing values
print(course_data.isnull().sum())
# Fill missing values (tranfsormation)
def transform_fill_programming_language(course_data):
    imputed = course_data.fillna({"programming_language": "R"})
    return imputed
transformed = transform_fill_programming_language(course_data)
# Get top 3 rated courses for each user
def transform_recommendations(avg_course_ratings, courses_to_recommend):
    # Merge both DataFrames
    merged = courses_to_recommend.merge(avg_course_ratings, on='course_id') 
    # Sort values by rating and group by user_id
    grouped = merged.sort_values("rating", ascending=False).groupby("user_id")
    # Produce the top 3 values and sort by user_id
    recommendations = grouped.head(3).sort_values("user_id").reset_index()
    final_recommendations = recommendations[["user_id", "course_id","rating"]]
    # Return final recommendations
    return final_recommendations
recommendations = transform_recommendations(avg_course_ratings, courses_to_recommend)
# Load the recommendations to a PostgreSQL table
recommendations.to_sql("recommendations", db_engine, if_exists='append')
# Define the ETL function
def etl():
    courses = extract_course_data(db_engines)
    rating = extract_Rating_data(db_engnines)
    courses = transform_fill_programming_language(courses)
    avg_course_ratings = transform_avg_rating(rating)
    courses_to_recommend = transform_courses_to_recommend(rating, courses)
    recommendations = transform_recommendations(avg_course_rating, courses_to_recommend)
    load_to_dwh(recommendations, db_engine)
# Create the DAG to execute the ETL daily
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
dag = DAG(dag_id='recommendations', schedule_interval="0 0 * * *")
task_recommendations = PythonOperator(task_id="recommendations_task", python_callable=etl, dag=dag)
# Try querying the recommendations table loaded in the Data Warehouse   
def recommendations_for_user(user_id, threshold=4.5):
    # Join with the courses table
    query = """
    SELECT title, rating FROM recommendations
    INNER JOIN courses ON courses.course_id = recommendations.course_id
    WHERE user_id=%(user_id)s AND rating>%(threshold)s
    ORDER BY rating DESC
    """
    # Add the threshold parameter
    predictions_df = pd.read_sql(query, db_engine, params = {"user_id": user_id, 
                                                             "threshold": threshold})
    return predictions_df.title.values
# Try the function you created
print(recommendations_for_user(12, 4.65))


#%% STREAMLINED DATA INGESTION WITH PANDAS

# Ch1: IMPORTING DATA FROM FLAT FILES

# Flat files: simple, esay-to-produce format, data stored as plain text (no formatting),
# one row per line, values for different fields are separated by a delimiter

# The pandas function read_csv() is used to read ALL flat files, whether csv or not

# Define the file's delimiter
import pandas as pd
tax_data = pd.read_csv('us_tax_data_2016.tsv', sep="\t") # here a tab

# LIMIT THE AMOUNT OF DATA READ FROM THE FLAT FILE
# - Choose only certain columns
# The keyword arg "usecols" can take a list of col names, col numbers, or functions (
# such as choosing all cols that start with 'a')
col_names = ['col1', 'col2']
df = pd.read_csv('file.csv', usecols=col_names)
col_numbers = ['0', '1']
df = pd.read_csv('file.csv', usecols=col_numbers)
# Limit the number of rows imported from a csv
df = pd.read_csv('file.csv', nrows=1000)
# Process a file in chunks
# 'skiprows' accepts a list of row numbers, a number of rows, or a function to filter rows, to skip
# You should set 'header' = None so that pandas knows there are no column names
# Eg get rows 1000 to 1500
df = pd.read_csv('file.csv', nrows=500, skiprows=1000, header=None)
# Assign column names when reading a csv (the list must be as long as the number of cols)
col_names = list(df_first_1000) # save col names from the first chunk
df_next500 = pd.read_csv('file.csv', nrows=500, skiprows=1000, header=None, names=col_names)

# HANDLING ERRORS AND MISSING DATA (eg. when files are not completely clean)
# Specifying column data types
df = pd.read_csv('file.csv', dtype={"zipcode": str}) # the rest of the types are inferred by pandas
# Specify which symbol stands for a missing value
# 'na_values' accepts a single value, a list of values, or a dictionary of column-value pairs
df = pd.read_csv('file.csv', na_values={"zipcode": 0}) # here 0s in zipcodes are NAs
# Deal with lines with errors (eg having more values than columns)
df = pd.read_csv('file.csv', error_bad_lines=False # skips unparseable records
                        warn_bad_lines=True) # display a message when a line is skipped


# Ch3: IMPORTING DATA FROM EXCEL FILES

# Unlike flat files, it can have formulas with updating results, and formatting

# Read an excel file with pandas
df = pd.read_excel('file.xlsx')

# SELECT COLUMNS AND ROWS FROM AN EXCEL FILE
# nrows: same as read_csv
# skiprows: number of rows, or specific row numbers
# usecols: name, positional number, or column letter
df = pd.read_excel('file.xlsx', skiprows=2, usecols='W:AB, AR')

# GETTING DATA FROM MULTIPLE EXCEL SHEETS
# Choose sheet: 'sheet_name': by names or zero-indexed position numbers
df = pd.read_excel('file.xlsx', sheet_name=0)
df = pd.read_excel('file.xlsx', sheet_name='first_sheet')
df = pd.read_excel('file.xlsx', sheet_name=['first', 'second'])
df = pd.read_excel('file.xlsx', sheet_name=[0, 'second'])
# Load all sheets of an Excel at once
df = pd.read_excel('file.xlsx', sheet_name=None) # it returns a dictionary
# Eg load all sheets into the same data frame
all_responses = pd.DataFrame()
survey_responses = pd.read_excel('file.xlsx', sheet_name=None)
for sheet_name, frame in survey_responses.items():
    frame['SheetID'] = sheet_name
    all_responses = all_responses.append(frame)

# SPECIAL CONSIDERATIONS WHEN READING BOOLEAN VARIABLES



# Ch3: IMPORTING DATA FROM DATABASES


#%% UNIT TESTING FOR DATA SCIENCE IN PYTHON

# A "unit" is any small independent piece of code (like a Python function or class)

# Ch1: UNIT TESTING BASICS

# Testing a custom function on the Python interpreter (eg. passing it the possible arguments and checking the result) is very inefficient.
# This is because the life cycle of a function involves several testing iterations.
# A function is tested after the first implementation and then any time the function is modified, which happens mainly when new bugs are found, new features are implemented or the code is refactored.
# UNIT TESTS automate the repetitive testing process and saves time.
# In this course I will write a complete unit test for a sample project.

# PYTEST: most popular Python library for unit testing
# Write a simple unit test using pytest
# Example data file "housing_data.txt": contains data on housing area (sq. ft.) and market price (usd)
# 1) Create a file called "test_row_to_list.py" ("row_to_list" is the name of the sample function)
# - When Pytest sees a file that starts with "test_" it understands that the file contains a unit test.
# - These files for unit tests are also called "test modules".
# 2) In the test module,
import pytest
import row_to_list
# The unit test will be written as a Python function whose name starts with "test_" (just like the test module)
def test_for_clean_row():
    assert row_to_list("2,081\t314,912\n") == ["2,081", "314,912"]
# The unit test usually corresponds to exactly one entry in the argument and return-value table for "row_to_list".
# It checks whether the function has the expected return value when called on this particular argument.
# The particular value will be using first ("2,081", "314,912") is a clean row.
# The actual check is done via an assert statement. Every test must contain one.
# If the function has a bug, the assert statement will raise an assertion error.
# For the second value ("\t293,410\n") we will create a second function called test_on_missing_area (bcs the arg has a missing area data)
def test_on_missing_area():
    assert row_to_list("\t293,410\n") is None
# And for arguments missing the tab that separates area and price
def test_on_missing_tab():
    assert row_to_list("1,462238,765\n") is None
# To test whether row_to_list is functioning well at any point on its life cycle, we would just run "test_row_to_list.py".
# The standard way to to this is by writting this in the command line
pytest test_row_to_list.py
# (I have to check on my PC, but the usual way to run the .py from the command line is !pytest test_on_missing_area.py )

# Understanding test result report (this should be watched in the video)
# When it says 'collected X items", X is the amount of unit tests
# Then it ouputs the name of each module and a character indicating the result.
# - ".F." stands for failure (an exception was raised)
# - '.' means the unit test was passed
# The following section contains detail information about failed tests
# - The line raised by the exception is marked by ">"
# - The lines marked by "E" contain details on the exception
# - The line containing "where" displays any return values that were calculated when running the assert statement
#       Here you can see the mismatch between the return value and the expected value
# The final line is a test result summary (hoy many tests passed, how many failed, and the time it took to run)

# More benefits of unit testing:
# - It serves as documentation.
# - Increase trust in a package as users can run them an verify that the package works
        # There are certain "badges" that a package may have which indicate
        # if is has passes a certain unit test (like "Azure Pipelines")
# - Can reduce downtime for a productive system. "Continuos integration": runs
    # all unit tests each time a new code is sent into production

# A "unit" is any small independent piece of code (like a Python function or class)
# They are called unit tests because they only test a unit

# An INTEGRATION TEST chekcs if multiple units work well together when they are
# connected and not independently

# END-TO-END check the whole software at ounce. They start from one end (eg. an
# unprocessed data file), go through all units till the other end, and check
# whether we get the correct result.


#%% WEB SCRAPING IN PYTHON

# Ch1: INTRODUCTION TO HTML

# WEB SCRAPING PIPELINE
# 1) SETUP: define the goal, identify the sources
# 2) ACQUISITION: accesing, parsing, extracting the data into meaninful data structures
# 3) PROCESSING: run data rhough analysis to achieve the desired goal

# This course focuses on 1), and uses Python's framework SCRAPY

# HTML: HyperText Markup Language
# Read by web browsers to render and display website content

# The elements contained between angled brackets <> are called "HTML TAGS"
# They usually come in pairs: a starting tag and a stopping tag with
#   a forward slash: <p> Thanks for watching </p>

# ROOT TAG: contains the main html content, <html> ... </html>
# BODY TAG: defines hte body of the html, <body> ... </body>    
# DIV TAG: defines the sections of the body, <div> ... </div>
# P TAGS: define paragraphs within the body <p> ... </p>

# These tags are usually nested, which gives rise to a hierarchy in the HTML
# The vocabulary used to describe this hierarchy is the same of a family tree: 
# as you move from left to right, you move "forward generations"; top to bottom
# "between the same generation", and moving "between siblings" is the elements
# come from the same parent element

# Example of HTML code (as string in Python):
html = '''
<html>
  <head>
    <title>Intro HTML</title>
  </head>
  <body>
    <p>Hello World!</p>
    <p>Enjoy DataCamp!</p>
  </body>
</html>
'''

# HTML ATTRIBUTES
# Tags can sometimes contain attributes which provide special instructions for 
# the contents contained within that tag.
# ATTRIBUTE NAMES are followed by an equal sign "=" and information which is 
# being passed to that attribute within the tag. The infromation should be in quotes.
# EXAMPLE: a "div" tag with attributes "id" and "class"
html = '''
<div id="unique-id" class="some class">
    ... div element contents
</div>
'''
# ID: a unique identifier for the tag element (in this case, the id "unique id
# should only belong to this specific element")
# CLASS: does not need to be unique
# No tag needs to have an ID nor a class, but all can be given them.
# A tag can belong to multiple classes, between "" and divided by spaces (in
# fact, in the example the tag belongs two two classes: "some" and "class")

# "A" TAG: a tag for hyperlinks
html = '''
<a href="https://www.datacamp.com">
    This text links to DataCamp
</a>
        '''
#  The href attribute is used to input the URL


# CRASH COURSE IN XPATH
# Now the idea is to turn wording navigation of html into a variable for the
# computer to ingest
# XPATH NOTATION: one of two common choices for the purpose above
# example:
xpath = '/html/body/div[2]'
# It uses a SINGLE FORWARD SLASH (in an analogous way as URLs) to move
# forward one generation, somethin like directories.
# The BRACKETS are used to specify which element/s we want to direct to.
# For example, there could be several "div" elements which could be children of
# the "body" element, that is, several "div" siblings. So you use the brackets
# to chose the "div" we want among those. In the example, we woulb be referring
# to the second div child of the body element
# Had it been written '/html/body/div', you would be referring to all div
# elements in that body element
# DOUBLE FORWARD SLASH
xpath = '//table'
# It tells us to look forward all future generations instead of one generation
# like the single forward slash.
# In the example, we aredirecting to ALL table elements of ANY HTML
# In the next example, navigate to all table elements which are descendants of
# the second div element:
xpath = '/html/body/div[2]//table'

# Exercise: navigate to the p element "Where am I"
html = '''
<html>
  <body>
    <div>
      <p>Good Luck!</p>
      <p>Not here...</p>
    </div>
    <div>
      <p>Where am I?</p>
    </div>
  </body>
</html>
'''
answer = '/html/body/div[2]/p'

# Exercise: Although we haven't yet gone deep into XPath, one thing we can do is
# select elements by their attributes using an XPath. For example, if we want to
# direct to the div element within the HTML document whose id attribute is "uid",
# then we could write the XPath string '//div[@id="uid"]'. The first part of
# this string, //div, first looks at all div elements in the HTML document.
# Then, using the brackets, we specify that we want only the div element with a
# specific id attribute (in this case uid). To note, the phrase @id="uid" in the
# brackets would be read as "attribute id equals uid".
# In this exercise, you will select all span elements whose class attribute
# equals "span-class". (Note: span is just another possible tag-name).
answer = '//span[@class="span-class"]'
# Direct to an ATTRIBUTE in HTML with '@'
xpath = '/html/body/div/p[2]/@class' # here we direct to the attribute "class"
# of the second paragraph element in the div elements of the body


# Ch2: XPATHS AND SELECTORS

xpath = '//p[1]'
# This would select the first sibling of all p elements (not very usual)

# WILDCARD CHARACTER '*': indicates we want to IGNORE the tag type
# For example, in the next code, we would select all children of the body
# element regardless of their type
xpath = '/html/body/*'

# The number of elements selected with the XPath string xpath = "/html/body/* is
# equal to the number of children of the body element; whereas the number of
# elements selected with the XPath string xpath = "/html/body//*" is equal to
# the total number of descendants of the body element.

# The number of elements selected by the XPath string xpath = "/*" is equal to
# the number of root elements within the HTML document, which is typically the 1
# html root element.

# The number of elements selected by the Xpath string xpath = "//*" is equal to
# the total number of elements in the entire HTML document.
# Select ALL elements in the HTML document:
xpath = "//*"

# Exercise: Create an XPath string to the desired paragraph element
html = '''
<html>
  <body>
    <div>
      <p>Hello World!</p>
      <div>
        <p>Choose DataCamp!</p>
      </div>
    </div>
    <div>
      <p>Thanks for Watching!</p>
    </div>
  </body>
</html>
'''
answer = '/html/body/div[1]/div/p'

# Select all elements of a specific class
xpath = '//p[@class="class-1"]'
# Select all elements of two specific classes
xpath = '//p[@class="class-1 class-2"]'
# Here, we select all p elements of class "class-1"
xpath = '//*[@id="uid"]'
# Here, we select all elements, of any tag type, that has id equal to "uid"
xpath = '//div[@id="uid"]/p[2]'
# Here, the second paragraph child of all div elements of id "uid"

# CONTAINS FUNCTION:
# syntax: contains(@attri-name, "string-expr")
# The function searches the attributes of that specific attribute name and 
# matches those where the string expression is a substring of the full attribute
# For example, the following expression chooses all elements in which the string
# "class-1" is contained as a substring within the full class attribute. It
# could retrieve, for example, the following elements:
html = '''
<p class="class-1">...</p>
<p class="class-1 class-2">...</p>
<p class="class-12">...</p>
'''
# Exercise: Fill in the blanks below to assign an XPath string to the variable
# xpath which directs to all href attribute values of the hyperlink a elements
# whose class attributes contain the string "package-snippet".
xpath = '//a[contains(@class,"package-snippet")]/@href'

# SCRAPY's SELECTOR OBJECTS:
# the scrapy object used to select portions of the html using xpath or
# CSS Locators
from scrapy import Selector
html = '''
<html>
  <body>
    <div class="hello datacamp">
        <p>Hello World!</p>
    </div>
    <p>Enjoy DataCamp!</p>
  </body>
</html>
'''
sel = Selector(text=html) # selector object 
# We can use the xpath call within a Selector to create new Selectors of 
# specified pieces of the html code. The return value is a SelectorList: a list
# with some Scrapy extras containing new selector objects.
# Example: select all paragraph objects:
sel.xpath('//p')
# The return value is a SELECTOR LIST, a list with some scrapy extracts
# containing new selector objects. In this case it would be two selector objects
# one for each paragraph.

# Extracting data from a SELECTOR LIST
sel.xpath('//p').extract()
# Get only the first element
sel.xpath('//p').extract_first()

# Chaining selectors
# This
sel.xpath('/html/body/div[2]')
# equivalent to this
sel.xpath('/html').xpath('./body/div[2]')
# and this
sel.xpath('/html').xpath('./body').xpath('./div[2]')

# Calculate the total number of children of an element
len( element.xpath('./*') )
# Get the total amount of elements in an html
len( sel.xpath('//*') )

# INSPECTING A REAL HTML
# View the HTML source code in Firefox: right click, view page source
# INSPECTING AN ELEMENT: select an element, right click, inspect element

# Get the HTML as a string (then we will see how to do this with scrapy only)
import requests
url = 'https://es.wikipedia.org/wiki/Web_scraping'
html = requests.get(url).content
sel = Selector(text=html)


# Ch3: CSS LOCATORS, CHAINING, AND RESPONSES

# CSS: cascading style sheets
# It describes how elements are displayed on the screen.
# It is an alternative to xpaths. Here are the transalations:
# - The '/' is replaced by '>' to move one generation forward, and there is no
# need for a / in the first character:
xpath = '/html/body/div' #equals
css = 'html > body > div'
# // is replaced by a blank space (except in the first character)
xpath = '//div/span//p' # equals
css = 'div > span p'
# [N] is replaced by :nth-of-type(N)
xpath = '//div/p[2]' # equals
css = 'div > p:nth-of-type(2)'
# Another example
xpath = '/html/body//div/p[2]' # equals
css = 'html > body div > p:nth-of-type(2)'

# ATTRIBUTES IN CSS: CSS has a simple notation for selecting elements by class or ID
# Find an element by class: tag_name.class_name
'p.class-1'
# Find an element by ID: tag_name#id_name
'div#uid'
# Example: CSS locator that navigates to the 'uid' div element, and then whichever
# paragraph of class1
css_locator = 'div#uid > p.class1'
# Example: select all elements whose class attribute belongs to class1
css_locator = '.class1'
# ! Differently from xpath, that locator matches elements which have class1 and
# other classes as well. For xpath to do this we used the contains() function.
# The CSS Locator string '*' selects all elements in the HTML document. 
# The CSS Locator string '*#uid' selects the element with id attribute equal to
# uid, but this is unnecessary since the string '#uid' will also do the same job.

# SELECT ATTRIBUTES with CSS
xpath = '//div[@id="uid"]/a/@href'
css = 'div#uid > a::attr(href)'

# TEXT EXTRACTION
html = '''
<p id="p-example">
    Hello World!
    Try <a href="http://www.datacamp.com">DataCamp</a> today!
</p>
'''
from scrapy import Selector
sel = Selector(text=html)
# In Xpath
sel.xpath('//p[@id="p-example"]/text()').extract() # The '/' before the text
# method refers us to all the chunks of text within that element, but not within
# future generations. If you use '//' you will point all chunks within that 
# element and within its descendants:
sel.xpath('//p[@id="p-example"]//text()').extract()
# In CSS, the first case ould be
sel.css('p#p-example::text').extract()
# And the second
sel.css('p#p-example ::text').extract()

# More examples of translations
xpath = '/html/body/span[1]//a'
css_locator = 'html > body > span:nth-of-type(1) a'
#
xpath = '//div[@id="uid"]/span//h4'
css_locator = 'div#uid > span h4'
# Create the CSS Locator to all children of the element whose id is uid
css_locator = '*#uid > *'
#
hrefs_from_css = course_as.css( '::attr(href)' )
hrefs_from_xpath = course_as.xpath( './@href' )
#
css_locator = 'p#p3::text'  
xpath = '//p[@id="p3"]/text()'

# RESPONSE OBJECTS in Scrapy
# The Response object has all the tools we learned in Selectors: xpaths and css
# methods followed by extract() and extract_first() (actually the Selector
# objects were an introduction to Response objects)
# The advantage is that on top of this, these objects keeps track of which url
# the html is from and hence gives us a tool to crawl between links on sites and
# scrape multiple links automatically. 
# The URL is stored as a string:
response.url
# FOLLOW METHOD: the response lets us follow a new link
response.follow(next_url)

# SCRAPING AN EXAMPLE SITE (DATACAMP)
# Scrape the course directory of DataCamp and create a list of link to the course
# pages.
import requests
url = 'https://app.datacamp.com/learn/courses'
# imagine there is a pre-loaded response
course_divs = response.css('div.course-block')
# amount of courses retreieved
print(len(course_divs)) 
# inspecting the first course
first_div = course_divs[0]
children = first_div.xpath('./*')
first_child = children[0]
print(first_child.extract()) # 
second_child = children[1]
print(second_child.extract()) # footer of the course block
third_child = children[2]
print(third_child.extract()) # specific info of the course block (invisible)
# Two options to create the list of courses:
# 1) Use a single CSS locator
links = response.css('div.course-block > a::attr(href)').extract()
# 2) Do it stepwise
course_divs = response.css('div.course-block')
hrefs = course_divs.xpath('./a/@href')
links = hrefs.extrect()
#
# Create a SelectorList of the course titles
crs_title_els = response.css('h4::text')
# Extract the course titles 
crs_titles = crs_title_els.extract()
# Print out the course titles 
for el in crs_titles:
  print( ">>", el )


# Ch4: SPIDERS

# General form of the code:
import scrapy
from scrapy.crawler import CrawlerProcess
    # Spider class. It must take scrapy.Spider as input to inherit all the methods from the scrapy.Spider class
class SpiderClassName(scrapy.Spider):
    name = "spider_name"
    # the code for your spider
    ...
    # Run the spider
process = CrawlerProcess()
process.crawl(SpiderClassName)
process.start()

# Example spider to scrape the DataCamp course directory
# Create the spider class, an pass the argument "scrapy.Spider" so that it
# inherits the methods from a scrapy spider
class DCspider(scrapy.Spider): # you can call the class anything you want

    name = "dc_spider" # here you can assign any string, but it is more neat to be similar to the class name

    # Method to define which site or sites to scrape, and where to send the information to be parsed
    def start_requests(self):
        urls = ['https://www.datacamp.com/courses/all']
        for url in urls:
            yield scrapy.Request(url = url, callback = self.parse)

    # Method to take the html code and simply write it to a file
    def parse(self, response):
        # simple example: write out the html
        html_file = 'DC_courses.html'
        with open(html_file, 'wb') as fout:
            fout.write(response.body)


# START REQUESTS method
def start_requests(self): # it cannot have a different name than this
    urls = ['https://www.datacamp.com/courses/all']
    for url in urls:
        yield scrapy.Requests(url=url, callback=self.parse) # send a response 
        # variable preloded with the HTML code from the url argument to the
        # parsing function defined in the 'callback' argument (the method "parse")
        # In other words, it sends a scrapy response object to a parsing function for processing.
        # With the scrapy.Request, many things happen under the hood
def parse(self, response): # response is the variable passed from the scrapy.Request call
# simple example: write out the html
    html_file = 'DC_courses.html'
    with open(html_file, 'wb') as fout:
        fout.write(response.body)

# PARSE method
# First example: extracts the links from the DataCamp course directory and
#   saves these links to a file with one link per line
class DCspider(scrapy.Spider):
    name = "dc_spider"

    def start_requests(self):
        urls = ['https://www.datacamp.com/courses/all']
        for url in urls:
            yield scrapy.Request(url = url, callback = self.parse)

    def parse(self, response):
        links = response.css('div.course-block > a::aattr(href)').extract()
        filepath = 'DC_link.csv'
        with open(filepath, 'w') as f:
            f.writelines( [link + 'n' for link in links] )

# Second example: crawl between more than one site. First weextract the course
#   links. Then, instead of printing them to a file, we make the spider follow
#   those links and parse those new sites in a second parser.
class DCspider(scrapy.Spider):
    name = "dc_spider"

    def start_requests(self):
        urls = ['https://www.datacamp.com/courses/all']
        for url in urls:
            yield scrapy.Request(url = url, callback = self.parse)

    def parse(self, response):
        links = response.css('div.course-block > a::aattr(href)').extract()
        for link in links:
            yield response.follow(url=link, callback=self.parse2)
            # instead of using scrapy.Request, we use the 'follow' method in the
            #   response varaible itself
    
    def parse2(self, response):
        # parse the course sites here

# In this exercise, we have set up a spider class which, when finished, will retrieve the author names from a shortened 
# version of the DataCamp course directory. The URL for the shortened version is stored in the variable url_short. 
# Your job will be to create the list of extracted author names in the parse method of the spider
import scrapy
# Create the Spider class
class DCspider( scrapy.Spider ):
  name = 'dcspider'
  # start_requests method
  def start_requests( self ):
    yield scrapy.Request( url = url_short, callback = self.parse )
  # parse method
  def parse( self, response ):
    # Create an extracted list of course author names
    author_names = response.css( 'p.course-block__author-name::text' ).extract()
    # Here we will just return the list of Authors
    return author_names

#This will be your first chance to play with a spider which will crawl between sites (by first collecting links from one
#  site, and following those links to parse new sites). This spider starts at the shortened DataCamp course directory, 
# then extracts the links of the courses in the parse method; from there, it will follow those links to extract the 
# course descriptions from each course page in the parse_descr method, and put these descriptions into the list 
# course_descrs. Your job is to complete the code so that the spider runs as desired!
import scrapy
# Create the Spider class
class DCdescr( scrapy.Spider ):
  name = 'dcdescr'
  # start_requests method
  def start_requests( self ):
    yield scrapy.Request( url = url_short, callback = self.parse )
  
  # First parse method
  def parse( self, response ):
    links = response.css( 'div.course-block > a::attr(href)' ).extract()
    # Follow each of the extracted links
    for link in links:
      yield response.follow(url=link, callback=self.parse_descr)
      
  # Second parsing method
  def parse_descr( self, response ):
    # Extract course description
    course_descr = response.css( 'p.course__description::text' ).extract_first()
    # For now, just yield the course description
    yield course_descr


# CAPSTONE: create an entire spider from start to finish which will
# - Collect all course links from the DataCamp directory
# - Follow those links to extract the title and the title of the course chapters.
# - Store the info in a dictionary
# General structure
import scrapy
from scrapy.crawler import CrawlerProcess

class DC_Chapter_Spider(scrapy.Spider):

    name = 'dc_chapter_spider'

    def start_request(self):
        url = 'https://www.datacamp.com/courses/all'
        yield scrapy.Request(url=url, callback=self.parse_front)

    ## Method to parse the front courses page
    def parse_front(self, response):
        # Narrow in on the course blocks
        coruse_blocks = response.css('div.course-block')
        # Direct to the course links
        course_links = course_blocks.xpath('./a/@href')
        # Extract the links (as a list of strings)
        links_to_follow = course_links.extract()
        # Follow the links to the next parser
        for url in links_to_follow:
            yield response.follow(url=url, callback=self.parse_pages)

    ## Method to parse course pages
    def parse_pages(self, response):
        # Direct to the course title text
        crs_title = response.xpath('//h1[contains(@class, "title")]/text()')
        # Extract and clean the course title text
        crs_title_ext = crs_title.extract_first().strip()
        # Direct to the chapter titles text
        ch_titles = response.css('h4.chapter__title::text')
        # Extract and clean the chapter titles text
        ch_titles_ext = [t.strip() for t in ch_titles.extract()]
        # Store this in a dictionary with course titles as keys and chapter titles as values
        dc_dict[crs_title_ext] = ch_titles_ext

# Initialize the dictionary **outside** of the Spider class
dc_dict = dict()

# Run the Spider
process = CrawlerProcess()
process.crawl(DC_Chapter_Spider)
process.start()


#%% MACHINE LEARNING FOR TIME SERIES DATA IN PYTHON

#  Covered: Feature extraction, model fitting, prediction and validation

# Ch1: TIME SERIES AND MACHINE LEARNING PRIMER

# Create an index for time

Time Series as Inputs to a Model


#%% END OF DATA CAMP ----------------------------------------------------------

# %% USEFUL COMMANDS

# Create a boolean mask that identifies the categorical columns of a data frame
categorical_mask = (df.dtypes == 'object')

# Creating a DataFrame from multiple equal-length lists
import pandas as pd
pd.DataFrame(list(zip(list1,list2), columns=['list1', 'list2']))

# Verification/ verify a condition: assert statement, returns nothing if a condition is met, and an error otherwise
assert True==False
assert True==True

# A function that returns True if a string if found in another string
import re
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)

    if match:
        return True
    return False

# Transform years into decades
import numpy as np
(np.floor(df['year']/10)*10).astype(np.int64)

## METODOS DE STRINGS
# Reemplazar una parte de un string por otra
saludo = 'hola'
print(saludo.replace('la', 'mbre'))

# Cambiar todo a mayuscula
place = "poolhouse"
print(place.upper())

# Cambiar la primera letra a mayuscula
print('hola'.capitalize())

## METODOS DE LISTAS
# Contar la cantidad de veces que aparece un elemento en una lista
lista1 = [1,2,3,4,1,1]
print(lista1.count(1))

# Obtener el indice de un objeto dentro de una lista / obtener la posicion de un elemento en una lista
lista1 = [1,2,3,4]
print(lista1.index(3))

# The ; sign is used to place commands on the same line. The following two code chunks are equivalent:
# Same line
print('Hello'); print('Bye')

# Separate lines
print('Hello')
print('Bye')

# Copiar objetos (listas)
lista1 = [1,2,3,4]
lista2 = lista1 # aca estas copiando la referencia a la lista1, no los objetos
del[lista2[2]]
print(lista1)
    # Para copiar los elementos, y no solo la referencia, hay que escribir
y = list[lista1]
# o
y = x[:]

# Eliminar un elemento de una lista
lista1 = [1,2,3,4]
del[lista1[2]]
print(lista1)

# Appendear / Agregar filas de otro data frame que no esten en el data frame actual
df_diff = df2[~df2.col1.isin(A.col1)]
df_full = pd.concat([df1, df_diff], ignore_index=True)

# Crear un archivo csv / Exportar un data frame a csv
df.to_csv("path")

# Cambiar el nombre de muchas columnas a la vez
cambio_cols = {'v1old': 'v1new', 'v2old': 'v2new'}
df.rename(columns=cambio_cols,
          inplace=True)

# Crear una variable fecha a partir de variables para anio mes y dia:
df['fecha'] = pd.to_datetime(dict(year=df.anio, month=df.mes, day=1)) # aca no tengo dia y le pongo 1

# Crear una columna / crear una variable en un data frame que tome diferentes valores de acuerdo una condicion
db['var'] = np.where(db['var2']=='valor', valorsitrue, valorsifalse)

# Crear una columna / crear una variable en un data frame que tome diferentes valores de acuerdo mas de una condicion
df = pd.DataFrame({'Type':list('ABBC'), 'Set':list('ZZXY')})
conditions = [
    (df['Set'] == 'Z') & (df['Type'] == 'A'),
    (df['Set'] == 'Z') & (df['Type'] == 'B'),
    (df['Type'] == 'B')]
choices = ['yellow', 'blue', 'purple']
df['color'] = np.select(conditions, choices, default='black')
print(df)

# Crear una columna variable con un promedio ponderado por grupo:
func_prom_pond = lambda x: np.average(x, weights=db.loc[x.index, "pesos"])
db['prom_pond'] = db4.groupby('grupo')['variable'].transform(func_prom_pond)

# Crear una muestra aleatorio de una secuencia
random.sample(range(1, 11), k=5)

# Generar una muestra aleatoria de un data frame
db.sample(n=1000, replace=False)

# Seleccionar las filas de un data frame que tengan missing value en una columna
df[df['var'].isnull()]

# Eliminar duplicados segun una variable:
df.drop_duplicates(subset=['var'])

# Merge / unir dos data frames
db1 = db1.merge(db2[['var_a_mergear']], how='left', on=[nombre de variable llave], indicator=True) #indicator te dice si agregar una columma que te diga el resultado del merge, ademas de true se le puede poner el strign que quieras

# Intertar una columna / variable al principio de un data frame
df.insert(0, 'nombrevar', var)

# Cambiar la posicion de una columna en un data frame (aca se pone en la posicion 0, al principio)
col = df.pop('Name')
df.insert(0, 'Name', col)

# Eliminar todos los objetos creados por el usuario:
for element in dir():
    if element[0:2] != "__":
        del globals()[element] # VERR PORQUE GENERA PROBLEMAS

# Seleccionar los elementos de una lsita que empiecen con determinado string:
result = [i for i in some_list if i.startswith('string')]

# Quedarse con / seleccionar las columnas de un tipo / clase determinado:
df.select_dtypes(np.number) # u otra clase, no probe con otra

# Convertir todas las columnas & variables a una clase determinada
def f(x):
    try:
        return x.astype(float) # o cualquier otra clase
    except:
        return x
df2 = df.apply(f)

# Quedarse con la primera fila de cada grupo
db.grouby('var').first()

# Ordenar un data frame segun multiples columnas:
df.sort_values(['var1', 'var2'], ascending=[False, True])

# Exportar un grafico a png o pdf o jpg:
import matplotlib as plt

# Calcular media / promedio por grupo:
db.groupby('variable', as_index=False).mean()
# y agregarlo como columna>
db['nueava_var'] = db0.groupby('var_grupo')['var_a_promediar'].transform('mean')

### Hacer graficos de carrera de barras:
# Primero hay que bajarse un programa para manejar videos que ese llama ffmpeg, unzipearlo en una carpeta tipo C o Program files, y agregarlo al path. Usar este tutorial:
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# Despues instalar el paquete de python para ese progrma en el cmd:
pip install ffmpeg-python
# Se puede chequear si se hizo bien poniendo en el cmd:
ffmpeg -version
# Instalar el paquete para hacer graficos de carreras en el cmd:
pip install bar_chart_race
# Ejemplo de como se usa:
# es importante que la base este en formato wide. Ver tutorial: https://www.dunderdata.com/blog/official-release-of-bar_chart_race-a-python-package-for-creating-animated-bar-chart-races
bcr.bar_chart_race(
    df = db2,
    filename = path_figures +  'carrera_1.mp4')

# Actualizar un paquete con pip. En cmd escribir:
pip install paquete --upgrade

# Contar los missing values de todas las columnas de un data frame:
print(df.isnull().sum())

# Eliminar la ultima fila o la primer fila de un data frame:
df.drop(df.tail(n).index,inplace=True) # drop last n rows
df.drop(df.head(n).index,inplace=True) # drop first n rows

# Eliminar la ultima columna de un data frame:
df.drop(df.columns[[-1,]], axis=1, inplace=True)

# Separar una cadena por un character:
texto.split(',') #en este caso es una coma
objeto.text.split('\n') #acá es si el objeto no es texto, primero lo pasás a texto

# Separar una cadena por comas, ignorando las comas entre comillas "":
funcion = re.compile(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")
funcion.split(objeto)

# Equivalente al comando "fillin" de Stata
import itertools
cols_a_combinar = ["var1", "var2", "var3"]
combinaciones = []
for var in cols_a_combinar:
    combinaciones.append(db[var].unique().tolist())
df1 = pd.DataFrame(columns = cols_a_combinar, data=list(itertools.product(*combinaciones)))
#df1


# Convertir un array (series) a una lista:
array.tolist()

# Obtener los valores únicos de una columna/variable:
db["variable"].unique()
#Si querés que sea una lista:
db["variable"].unique().tolist()

# Factorizar/Encodear una variable (asignarle a cada valor único de string un número):
pd.factorize(db['variable'])

# Cread dummies para diferentes valores de una misma variable (en realidad se dice "one-hot" encoding, dummies seria si dejas una categoria afuera, como en una regresión):           
pd.get_dummies(df, columns=["variable"])

# Eliminar todas las variables que empiezan con determinado string:
db = db.drop(db.filter(like='stringinicial').columns, axis=1)

# Eliminar una lista de columnas / variables a la vez:
eliminar = ['var1', 'var2']
db.drop(eliminar, axis=1, inplace=True)

# Tabular una variable:
db.groupby(['variable']).size()
#mejor:
db.variable.value_counts()

# Tabular una variable y quedarse con los procentajes de cada grupo:
db.groupby(['var1', 'var2'])['var3'].agg('count') / db['var3'].agg('count')

# Traductor de Stata a Python
http://www.danielmsullivan.com/pages/tutorial_stata_to_python.html

# Seleccionar / ver determinadas columnas de un data frame segun su posicion: (ej. ultimas 10 columnas)
df.iloc[:,-10:]

# Unir elementos/string/cadenas de una lista (o cualquier otro string/cadena) con un string:
" ".join(item for item in lista)

# Seleccionar / ver determinadas columnas de un data frame según su nombre:
db[['var1', 'var2']]

# Seleccionar filas de acuerdo al valor de una columa
db.loc[db['variable']==valor]

# Seleccionar filas de acuerdo a varios valores de una misma columna:
db[ db['variable'].isin(['valor1', 'valor2', 'valor3']) ]

#Abrir un CSV como dataframe con Pandas
import pandas as pd
df = pd.read_csv('archivo.csv')

#Cambiar un número de entero a no entero:
variable = float(variableentera)

#Convertir de no entero a entero
variable = int(variablefloat)

#Floored division: hace una división y te devuelve un núm entero si los dos son enteros y sino n float:
5//2

#OJO CON LA TOLERANCIA. A veces le preguntas si x==2.5 pero en realidad es 2.50000, o una movida así con el tema de los floats y pedirle enteros

#Hacer una lista:
nombrelista = [ ]
#(si le pones parentesis es un tupple, es inmutable)

#Llamar a un elemento de la lista (ojo que empieza desde la posición cero)
print(nombrelista[0])
#Para pedir el último valor ponés -1

#Llamar un subgrupo de la lista:
print(nombrelista[2:4])
#Ojo que el último (posición 4) no lo agarra

#Agregar un elemento a la lista: 
nombrelista.append(29)

# Loopear según el nombre de las variables (ej con un sufijo de tiempo) (aca las puse en una lista:
var1=1
var2=13
var3=-5
lista = []
for i in range(1,4):
    lista.append( eval("var"+str(i)) )
lista

#Un loop: agarrar elemento por elemento y mostrarlo:
for x in mylist:
    print(x)

#Un loop: agregar elementos a la lista, acá agrega el 4 y el 5
a = [1,2,3]
a += [4,5] 
#(o sin corchetes)

# Loopear por numeros:
for i in range(2,10,2): #inicio, fin, salto
    ...

#Quedarte con los strings de una lista:
    
# Imaginemos que tenemos una lista de nombres no ordenados que de alguna manera se incluyeron algunos números al azar.
# Para este ejercicio, queremos imprimir la lista alfabética de nombres sin los números.
# Esta no es la mejor manera de hacer el ejercicio, pero ilustrará un montón de técnicas.
names = ["John", 3234, 2342, 3323, "Eric", 234, "Jessica", 734978234, "Lois", 2384]
print("Number of names in list: {}".format(len(names)))
# Primero eliminamos esos números
new_names = []
for n in names:
    if isinstance(n, str):
        # Si n es string, agregar a la lista
        # Notar la doble sangría
        new_names.append(n)

#Eliminar un (dos) elemento de la lista:
lista[0:2]=[ ]

# # Find all files with a specific pattern
from glob import glob
files = glob('direc/direc2/*.csv')

 
#%% CURSO UNSAM

#Ejecutar en una terminal de Windows:
C:\SomeFolder>hello.py
hello world

C:\SomeFolder>c:\python36\python hello.py
hello world

#A veces es conveniente especificar un bloque de código que no haga nada. El comando pass se usa para eso.
if a > b:
    pass
else:
    print('No ganó a')

x + y      #Suma
x - y      #Resta
x * y      #Multiplicación
x / y      #División (da un float, no un int)
x // y     #División entera (da un int)
x % y      #Módulo (resto)
x ** y     #Potencia
abs(x)     #Valor absoluto


x << n     #Desplazamiento de los bits a la izquierda
x >> n     #Desplazamiento de los bits a la derecha
x & y      #AND bit a bit.
x | y      #OR bit a bit.
x ^ y      #XOR bit a bit.
~x         #NOT bit a bit.

import math
a = math.sqrt(x)
b = math.sin(x)
c = math.cos(x)
d = math.tan(x)
e = math.log(x)

x < y      #Menor que
x <= y     #Menor o igual que
x > y      #Mayor que
x >= y     #Mayor o igual que
x == y     #Igual a
x != y     #No igual a

#Con esto en mente, ¿podrías explicar el siguiente comportamiento?
>>> bool("False")
True
>>>

#Normalmente las cadenas de caracteres solo ocupan una linea. Las comillas triples nos permiten capturar todo el texto encerrado a lo largo de múltiples lineas:
# Comillas triples
c = '''
Yo no tengo en el amor
Quien me venga con querellas;
Como esas aves tan bellas
Que saltan de rama en rama
Yo hago en el trébol mi cama
Y me cubren las estrellas.
'''

#Código de escape
#Los códigos de escape (escape codes) son expresiones que comienzan con una barra invertida, \ y se usan para representar caracteres que no pueden ser fácilmente tipeados directamente con el teclado. Estos son algunos códigos de escape usuales:
'\n'      #Avanzar una línea
'\r'      #Retorno de carro El retorno de carro (código '\r') mueve el cursor al comienzo de la línea pero sin avanzar una línea. El origen de su nombre está relacionado con las máquinas de escribir.
'\t'      #Tabulador
'\''      #Comilla literal
'\"'      #Comilla doble literal
'\\'      #Barra invertida literal

#Indexación de cadenas
#Las cadenas funcionan como los vectores multidimensionales en matemática, permitiendo el acceso a los caracteres individuales. El índice comienza a contar en cero. Los índices negativos se usan para especificar una posición respecto al final de la cadena.
a = 'Hello world'
b = a[0]          # 'H'
c = a[4]          # 'o'
d = a[-1]         # 'd' (fin de cadena)
También se puede rebanar (slice) o seleccionar subcadenas especificando un range de índices con :.
d = a[:5]     # 'Hello'
e = a[6:]     # 'world'
f = a[3:8]    # 'lo wo'
g = a[-5:]    # 'world'

Operaciones con cadenas
Concatenación, longitud, pertenencia y replicación.
# Concatenación (+)
a = 'Hello' + 'World'   # 'HelloWorld'
b = 'Say ' + a          # 'Say HelloWorld'

# Longitud (len)
s = 'Hello'
len(s)                  # 5

# Test de pertenencia (in, not in)
t = 'e' in s            # True
f = 'x' in s            # False
g = 'hi' not in s       # True

# Replicación (s * n)
rep = s * 5             # 'HelloHelloHelloHelloHello'

#Métodos de las cadenas
#Las cadenas en Python tienen métodos que realizan diversas operaciones con este tipo de datos.
#Ejemplo: sacar (strip) los espacios en blanco sobrantes al inicio o al final de una cadena.
s = '  Hello '
t = s.strip()     # 'Hello'
#Ejemplo: Conversión entre mayúsculas y minúsculas.
s = 'Hello'
l = s.lower()     # 'hello'
u = s.upper()     # 'HELLO'
#Ejemplo: Reemplazo de texto.
s = 'Hello world'
t = s.replace('Hello' , 'Hallo')   # 'Hallo world'
s.center(3, '*') # agrega 3 asteriscos atras y adelante del texto. Si no pones ningun string agrega espacios
s.rjust(4) # justificacion a la derecha
s.ljust(4)

#Más métodos de cadenas:
#Los strings (cadenas) ofrecen una amplia variedad de métodos para testear y manipular textos. Estos son algunos de los métodos:
s.endswith(suffix)     # Verifica si termina con el sufijo
s.find(t)              # Primera aparición de t en s (o -1 si no está)
s.index(t)             # Primera aparición de t en s (error si no está)
s.isalpha()            # Verifica si los caracteres son alfabéticos
s.isdigit()            # Verifica si los caracteres son numéricos
s.islower()            # Verifica si los caracteres son minúsculas
s.isupper()            # Verifica si los caracteres son mayúsculas
s.join(slist)          # Une una lista de cadenas usando s como delimitador
s.lower()              # Convertir a minúsculas
s.replace(old,new)     # Reemplaza texto
s.split([delim])       # Parte la cadena en subcadenas
s.startswith(prefix)   # Verifica si comienza con un sufijo
s.strip()              # Elimina espacios en blanco al inicio o al final
s.upper()              # Convierte a mayúsculas

#Los strings son "inmutables" o de sólo lectura. Una vez creados, su valor no puede ser cambiado. Esto implica que las operaciones y métodos que manipulan cadenas deben crear nuevas cadenas para almacenar su resultado.

#Ejercicio 1.16: Testeo de pertenencia (test de subcadena)¶
#Experimentá con el operador in para buscar subcadenas. En el intérprete interactivo probá estas operaciones:
>>> 'Naranja' in frutas
?
>>> 'nana' in frutas
True
>>> 'Lima' in frutas
?
>>>
#Ejercicio 1.21: Expresiones regulares
#Una limitación de las operaciones básicas de cadenas es que no ofrecen ningún tipo de transformación usando patrones más sofisticados. Para eso vas a tener que usar el módulo re de Python y aprender a usar expresiones regulares. El manejo de estas expresiones es un tema en sí mismo. A continuación presentamos un corto ejemplo:
>>> texto = 'Hoy es 6/8/2020. Mañana será 7/8/2020.'
>>> # Encontrar las apariciones de una fecha en el texto
>>> import re
>>> re.findall(r'\d+/\d+/\d+', texto)
['6/8/2020', '7/8/2020']
>>> # Reemplazá esas apariciones, cambiando el formato
>>> re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\2-\1', texto)
'Hoy es 2020-8-6. Mañana será 2020-8-7.'
>>>
#Para más información sobre el módulo re, mirá la documentación oficial en inglés o algún tutorial en castellano. Es un tema que escapa al contenido del curso pero te recomendamos que mires en detalle en algún momento. Aunque no justo ahora. Sigamos...
#Comentario
#A medida que empezás a usar Python es usual que quieras saber qué otras operaciones admiten los objetos con los que estás trabajando. Por ejemplo. ¿cómo podés averiguar qué operaciones se pueden hacer con una cadena?
#Dependiendo de tu entorno de Python, podrás ver una lista de métodos disponibles apretando la tecla tab. Por ejemplo, intentá esto:
>>> s = 'hello world'
>>> s.<tecla tab>
>>>
#Si al presionar tab no pasa nada, podés volver al viejo uso de la función dir(). Por ejemplo:
>>> s = 'hello'
>>> dir(s)
['__add__', '__class__', '__contains__', ..., 'find', 'format',
'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
'title', 'translate', 'upper', 'zfill']
>>>
#dir() produce una lista con todas las operaciones que pueden aparecer luego del parámetro que le pasaste, en este caso s. También podés usar el comando help() para obtener más información sobre una operación específica:
>>> help(s.upper)
#Help on built-in function upper:

#upper(...)
    S.upper() -> string

    #Return a copy of the string S converted to uppercase.


#Los elementos de una cadena pueden ser separados en una lista usando el método split():
line = 'Pera,100,490.10'
row = line.split(',') #la coma indica el elemento que separa
row
['Pera', '100', '490.10']

#Para encontrar rápidamente la posición de un elemento en una lista, usá index().
nombres = ['Rosita','Manuel','Luciana']
nombres.index('Luciana')   # 2
#Si el elemento está presente en más de una posición, index() te va a devolver el índice de la primera aparición. Si el elemento no está en la lista se va a generar una excepción de tipo ValueError.
#rdenar una lista
#Las listas pueden ser ordenadas "in-place", es decir, sin usar nuevas variables.
s = [10, 1, 7, 3]
s.sort()                    # [1, 3, 7, 10]

# Orden inverso
s = [10, 1, 7, 3]
s.sort(reverse=True)        # [10, 7, 3, 1]

# Funciona con cualquier tipo de datos que tengan orden
s = ['foo', 'bar', 'spam']
s.sort()                    # ['bar', 'foo', 'spam']
#Usá sorted() si querés generar una nueva lista ordenada en lugar de ordenar la misma:
t = sorted(s)               # s queda igual, t guarda los valores ordenados

#Podés acceder a los elementos de las listas anidadas usando múltiples operaciones de acceso por índice.
>>> items[0]
'spam'
>>> items[0][0]
's'
>>> items[1]
['Banana', 'Mango', 'Frambuesa', 'Pera', 'Granada', 'Manzana', 'Lima']
>>> items[1][1]
'Mango'
>>> items[1][1][2]
'n'
>>> items[2]
[101, 102, 103]
>>> items[2][1]
102
>>>
#MANERA DE VER LO QUE ESTÁS ITERANDO (con un ejemplo de la clase):
for i,c in enumerate(cadena):
        capadepenapa=capadepenapa+c
        if c in ("aeiou"):
            capadepenapa=capadepenapa+"p"+c #es lo mismo que poner capadepenapa += "p"+c
        print(i,c,capadepenapa)
print(capadepenapa)


#PARA HACER UN BLOQUE/SECCIÓN:
####   #%% Sección 1

Cómo chequear la versión de Python:
import sys
print(sys.version)


# Cuantiles ponderados
 
def comando_cuantiles(df, variable, cuantiles, ponderador=None):

    import matplotlib.pyplot as plt
    #!pip install weightedcalcs
    import weightedcalcs as wc

    if ponderador!=None:
        calc = wc.Calculator(ponderador)
        percentiles = []
        #Computo los percentiles
        for x in range(1,cuantiles+1) :
            p = calc.quantile(df[df[variable]>0], variable,x/100)
            percentiles = percentiles + [p]

        data=df[df[variable]>0]
        lista_df = []
        link = []
        
        for index, row in data.iterrows():
            t = False
            per=0
            for i in percentiles:
                if t==False:   
                    if row[variable]>=i:
                        t=False
                    else:
                        t=True
                    per += 1
            lista_df = lista_df  + [per]
            link = link + [row['link']]
        dict_df = {'link':link,'percentil':lista_df}
        out = pd.DataFrame.from_dict(dict_df)
        out.percentil = out.percentil.astype(int)

        return out
    # else:
    #     df.quantile(q=cuantiles)
        
#     bar_df = gdf.groupby('percentil').agg({'ponderador':'sum'})
#     bar_df = bar_df.reset_index()
#     plt.bar(bar_df['percentil'], bar_df[ponderador])

# %% SPATIAL DATA

import geopandas as gpd

# Transform a geodataframe to a dataframe and drop the geometry column
df = pd.DataFrame(gdf.drop(columns='geometry'))

# Read a shape / geospatial file
mapa = gpd.read_file('path')

# Plot the map
mapa.plot()

# Explore the map / plot the map on a real map:
mapa.explore()

# Generate a distance matrix between points in a geo data frame:
matriz_distancias = mapa.geometry.apply(lambda g: mapa.distance(g))

# Convert a data frame to a geo data frame
db_gdf = gpd.GeoDataFrame(data=db, geometry=gpd.points_from_xy(db.longitude, db.latitude), crs='epsg:4326')

# Spatial joins / Select points within a polygon
db_gdf = gpd.sjoin(db_gdf, polydf, op = 'within')

# Change the CRS of a geo data frame
db_gdf = db_gdf.to_crs('epsg:4326') # I think there is another one called set_crs # and another is db_gdf.crs = {'init': 'epsg:4326'}

# Create a shapefile / convert a geo data frame to a shapefile
gdf.to_file("algo.shp")

# Difference between these two CRSs 3857 and 4326: https://gist.github.com/Rub21/49ed3e8fea3ae5527ea913bf80fbb8d7

# Plot two layers together
base_plot = poligono.plot()
puntos.plot(ax=base_plot, color='blue');

# See ipynb "arreglo_espacioal_estaciones_servicio"