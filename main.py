import numpy, pandas
from sklearn.cluster import MeanShift
from sklearn import preprocessing

'''
    @description: filters data from a data frame
'''


def filter_data(df, data_filter):
    return df[data_filter]


'''
    @description: converts data frame data to numeric data
'''


def handle_non_numerical_data(df):
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}

        def convert_to_int(val):
            return text_digit_values[val]

        # print(column,df[column].dtype)
        if df[column].dtype != numpy.int64 and df[column].dtype != numpy.float64:

            column_contents = df[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_values[unique] = x
                    x += 1
            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df


data_frame = pandas.read_excel('titanic.xls')

original_data_frame = pandas.DataFrame.copy(data_frame)

data_frame.drop(['body', 'name'], 1, inplace=True)

data_frame.fillna(0, inplace=True)

data_frame = handle_non_numerical_data(data_frame)

data_frame.drop(['ticket', 'home.dest'], 1, inplace=True)

X = numpy.array(data_frame.drop(['survived'], 1).astype(float))

X = preprocessing.scale(X)
y = numpy.array(data_frame['survived'])

classifier = MeanShift()
classifier.fit(X)

labels = classifier.labels_

cluster_centers = classifier.cluster_centers_

original_data_frame['cluster_group'] = numpy.nan

for i in range(len(X)):
    original_data_frame['cluster_group'].iloc[i] = labels[i]

number_of_clusters = len(numpy.unique(labels))
survival_rates = {}

for i in range(number_of_clusters):
    temp_data_frame = original_data_frame[ (original_data_frame['cluster_group'] == float(i)) ]

    # print(temp_data_frame.head())

    survival_cluster = temp_data_frame[ (temp_data_frame['survived'] == 1) ]

    survival_rate = len(survival_cluster) / len(temp_data_frame)

    # print(i, survival_rate)

    survival_rates[i] = survival_rate

# We can see below that many more passengers in cluster 1 survived than others 75% versus 0 :36 and 3 at 11%
print("Cluster:Survival rate: ", survival_rates)

print("Type of customers in cluster 1 with highest survival rate:\n", original_data_frame[ (original_data_frame['cluster_group']==1) ])

print("Type of customers in cluster 0 with lower survival rate:\n", original_data_frame[ (original_data_frame['cluster_group']==0) ].describe())

print("Type of customer in cluster 2 with lowest survival rate:\n", original_data_frame[ (original_data_frame['cluster_group']==2) ].describe())

# Lets get the first class in cluster 0. We can see that they have a much worse survival rate than those in cluster 1 which has a higher average fare
cluster_0 = (original_data_frame[ (original_data_frame['cluster_group']==0) ])
cluster_0_fc = (cluster_0[ (cluster_0['pclass']==1) ])
print("Cluster 0 first class passenger have a lower chance of survival but there fare was lower:\n", cluster_0_fc.describe())