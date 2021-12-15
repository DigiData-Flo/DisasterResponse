import sys
import joblib
import seaborn as sns
import re
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from feature_extraction.text_length_extractor import TextlengthExtractor
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, make_scorer, recall_score
from sklearn.metrics import accuracy_score


## Matplotlib Configuration
# Define Plot Color
blue = sns.color_palette()[0]
# Plot Style
plt.style.use('seaborn')
# Define Matplotlib Params
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (12, 9)
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['figure.titlesize'] = 'xx-large' 
mpl.rcParams['axes.titlesize'] = 'large' 






def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('labeled_messages', engine)
    category_names = df.columns[4:]
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    
    return X, y, category_names


def tokenize(text):
    
    
    stop_words = stopwords.words("english")


    # Step 1 Normalisation
    text = re.sub(r'[^A-Za-z0-9]', ' ', text.lower())
    
    # Step 2 Tokenzation
    words = word_tokenize(text)
    
    #Step 3 Remove stopwords
    words = [w for w in words if w not in stop_words]
    
    #Step 4 Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(word=w) for w in words]
    
    return tokens

def build_model(n_jobs=7):
    
    ## Define a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_length', TextlengthExtractor())
            
        ])),

        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    ## Define a scorer
    scorer = make_scorer(recall_score, average='micro')


    ## define paramaters for GridSearchCV
    parameters = {
        'features__text_pipeline__vect__min_df': [1, 5],
        'clf__estimator__min_samples_split': [2,10],
        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=11)

    return cv
    

def create_metric_dataframe(y_test, y_pred, labels):
    '''
    INPUT:
            Test labels dataframe: y_test, 
            Prediction dataframe: y_pred,
            Array of labels: labels
    OUTPUT: 
            Dataframe with index column, metric, ,etric_value, label
            index: True positive, or True negative
            metric: accuracy, precision, recall, fscore
            metric_value: the value for the specific metric
            label: the label for the prediction
    WHAT ITS DO:
            Use precision_recall_fscore and accuracy_score functions from sklearn.metrics
            Combine the metrics to a large metric dataframe
    '''
    
    ## Initiate an empty data frame with predefined columns
    ## The column index contains 0 or 1 because the precision recall fscore metrics are calculated 
    ## For true positive and true negative
    metric_complete = pd.DataFrame(columns=['index', 'metric', 'metric_value', 'label'])

    ## Loop over all labes an call the precision_recall_fscore method for each label
    for label in labels:
        metric = precision_recall_fscore_support(y_test.loc[:,label], y_pred.loc[:,label], zero_division=0)
        accuracy = accuracy_score(y_test.loc[:,label], y_pred.loc[:,label])
        ## Create a dataframe from label metric
        metric_df = pd.DataFrame({'precision': metric[0], 'recall': metric[1], 'fscore': metric[2], 'accuracy': accuracy})
        ## Melt the dataframe
        metric_df = metric_df.reset_index().melt(id_vars='index', var_name='metric', value_name='metric_value')
        ## Add Label column
        metric_df['label'] = label
        ## Concat each label dataframe with complete dataframe
        metric_complete = pd.concat([metric_complete, metric_df])


        metric_complete.sort_values(by=['metric_value'], ascending=False, inplace=True)
    
    return metric_complete

def create_metric_mean(metric_complete, index, order):
    '''
    INPUT: 
            metric dataframe: Dataframe with different 
            metrics in metric column and metric_value column
    OUTPUT: mean_value dataframe and the mean values itself in given order
    '''
    
    
    metric_means = metric_complete.groupby(['index','metric']).metric_value.mean().reset_index()

    mean_values_df = metric_means.query('index==@index')[['metric', 'metric_value']]
    mean_values_df.set_index('metric', inplace=True)
    mean_values_df = mean_values_df.reindex(order)
    mean_values = mean_values_df.metric_value.values
    
    return mean_values_df, mean_values


def evaluate_model(model, X_test, y_test, category_names):
    
    ## Create a prediction Dataframe on the test data
    y_pred = pd.DataFrame((model.predict(X_test)), columns=category_names)
    metric_order = ['accuracy', 'precision', 'recall', 'fscore']
    
    metric_complete = create_metric_dataframe(y_test, y_pred, category_names)
    metric_means, mean_values = create_metric_mean(metric_complete, 1, metric_order)
    
    ## Plot for recall
    ax = sns.barplot(data=metric_complete.query('index==1 and metric=="recall"'), x='metric_value', y='label', color=blue)

    for p in ax.patches:
        text = '{}'.format(round(p.get_width(),3))
        xy = (p.get_width(), p.get_y() + p.get_height() / 2)
        ax.annotate(text, xy, ha = 'left', va = 'center', xytext = (5, 0), textcoords = 'offset points')
    
    plt.show()

    ax = sns.barplot(data=metric_means.reset_index(), x='metric', y='metric_value', order=metric_order, color=blue)

    for p in ax.patches:
        text = '{}'.format(round(p.get_height(),3))
        xy = (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height())
        ax.annotate(text, xy, ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.show()
    



def save_model(model, model_filepath):
    joblib.dump(model, model_filepath, compress=1)





def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()