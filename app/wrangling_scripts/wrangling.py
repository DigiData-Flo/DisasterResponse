import pandas as pd
import plotly.graph_objs as go


def return_figures(df):
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    ## Category count
    ## From Database Database
    ## Bar Plot
    graph_one = []
    ## get categories from dataframe
    y = df.iloc[:,4:]
    ## drop child alone no trainingsd data
    y = y.drop('child_alone', axis=1)
    ## create dataframe with counts for each category
    category_count = y.sum().reset_index()
    ## Rename Columns
    category_count.columns = ['category', 'count']
    ## Sort the dataframe
    category_count = category_count.sort_values(by='count', ascending=False)

    count = category_count['count'].tolist()
    ## Extract one time values for x-axis year
    categories = category_count['category'].tolist()
    #print('count: ', len(count))
    #print('categories: ', len(categories))
    ## append each label data to graph_one list
    graph_one.append(
        ## type scatter
        go.Bar(
        y = count,
        x = categories ))

    ## Plot 1 layout
    layout_one = dict(
        title = 'Number of Messages for each Category',
        yaxis = dict(title = 'Message Count')
    )


    ## Detailed Recall plot
    ## From metric csv file from training process
    ## Bar Plot

    graph_two = []
    #ax = sns.barplot(data=metric_complete_3.query('index==1 and metric=="recall"'), x='metric_value', y='label', color=blue)
    metric_complete = pd.read_csv('../data/metric_complete.csv')
    recall_values = metric_complete.query('index==1 and metric=="recall"')['metric_value'].round(3)
    categories = metric_complete.query('index==1 and metric=="recall"')['label']

    graph_two.append(
        go.Bar(
        y = recall_values,
        x = categories ))

    ## Plot 2 layout
    layout_two = dict(
        title = 'Recall Metric for different categories',
        yaxis = dict(title = 'Recall Value')
    )


    ## Difference in Placement Plot
    ## From metric csv file from training process
    ## Bar Plot
    graph_three = []
    #sns.barplot(data=performance, x='performance', y='category', color=blue)
    performance = pd.read_csv('../data/performance.csv')
    difference = performance.performance.values
    categories = performance.category.values

    graph_three.append(
        go.Bar(
        y = difference,
        x = categories ))

    ## Plot 3 layout
    layout_three = dict(
        title = 'Difference between the placement according to the number of messages<br>and the placement according to the recall value',
        yaxis = dict(title = 'Difference to message count placement')
    )

    ## Metric Mean Plot
    ## From metric csv file from training process
    ## Bar Plot
    graph_four = []
    #ax = sns.barplot(data=metric_means_3.reset_index(), x='metric', y='metric_value', order=metric_order, color=blue)
    metric_mean = metric_complete.groupby(['metric', 'index']).mean().reset_index().query('index==1')
    metric_order = ['accuracy', 'precision', 'recall', 'fscore']
    metric = metric_mean['metric']
    metric_value = metric_mean['metric_value'].round(3)



    graph_four.append(
        go.Bar(
        y = metric_value,
        x = metric ))

    ## Plot 4 layout
    layout_four = dict(
        title = 'Mean Values for different Metrics over all Categories',
        yaxis = dict(title = 'Metric Mean Value')
    )






    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures
