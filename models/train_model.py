import sys
import joblib
import pandas as pd
import sqlite3
import time as time
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

from sklearn.metrics import f1_score, precision_score, recall_score


def load_data(database_filepath):
    """
    A function to load saved SQL database.

    Args:
        database_filepath
    Returns:
        X, y for model training
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    

    X = df.message
    y = df.drop(labels=['message','genre'],axis=1)

    return X, y

def process_text(text):
    """
    A function to process text for a single document by removing punctuation and capital letters,
    tokenizing the text and removing stop words.

    Args:
        text (str)
    Returns:
        words
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]'," ",text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    return words

    
def create_pipeline():
    """
    A function to create a ML pipeline composed of 3 estimators: CountVectorizer,
    TF-IDFTransformer and XGBClassifier for multi-output classification for each category.

    Returns:
        pipeline
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=process_text)),
            ('tfidf', TfidfTransformer()),
            ('clf',MultiOutputClassifier(
                estimator=xgb.XGBClassifier(random_state=42, 
                                            use_label_encoder=False,verbosity = 0)))
            ])
    
    parameters = {
    'clf__estimator__n_estimators': [150, 200],
    'clf__estimator__max_depth': [3, 4, 5],
    'clf__estimator__learning_rate': [0.05, 0.1, 0.2],
#    'clf__estimator__early_stopping_rounds': [5,10],
    }

    # GridSearchCV takes a long time to run (~1.8hrs) so is commented out:
    # tuned_classifier = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=1)
    # tuned_classifier.best_estimator_ #results: learning_rate=0.1, max_depth=4, n_estimators=200,
    # pipeline = tuned_classifier
    
    return pipeline

def evaluate_model(y_test, y_pred):
    """
    A function to combine the sum of identified categories, f-score and precison/ recall scores to a
    dataframe for each category.

    Args:
        y_test, y_pred
    Returns:
        eval_df (dataframe)
    """

    data = {'Truth':  y_test.sum(axis=0),
            'Prediction': y_pred.sum(axis=0),
            }
    eval_df = pd.DataFrame(data)
    
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    f1_scores = []
    prec_scores = []
    recall_scores = []
    for col in y_pred.columns:
        f1_scores.append(f1_score(y_test[col], y_pred[col]))
        prec_scores.append(precision_score(y_test[col], y_pred[col]))
        recall_scores.append(recall_score(y_test[col], y_pred[col]))

    eval_df['f1_score'] = f1_scores
    eval_df['precision_score'] = prec_scores
    eval_df['recall_score'] = recall_scores
    eval_df = eval_df.sort_values(by='f1_score',ascending=False)
    
    #saving evaluation metric df for use in app (uncomment for 1st model training only):
    # Note this is not live --> instead app uses a saved version currently in the directory
    # with open('eval_df.pkl', 'wb') as f:
    #     pickle.dump(eval_df, f) 
    #print ('F1-score for each category: \n', eval_df['f1_score'])
    
    return 

def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)
    return


def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, y = load_data(database_filepath)

        pipeline = create_pipeline()
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 22)

        #This code is needed for the app to display the predicted categories in the correct order:
        ######
        # load evaluation metrics from previous model training
        eval_df = pickle.load(open('models/eval_df.pkl', 'rb'))
        # reorder categories by f-score from existing trained model --> so y_pred is in correct order for app display
        new_order = list(eval_df.sort_values(by='f1_score',ascending=False).iloc[:,1].reset_index()['index'])        
        y_train = y_train.reindex(columns=new_order)
        ######
        
        print('Training model...')
        starttime = time.time() 
        pipeline.fit(X_train, y_train)
        print ('\n    Model trained in secs: ', time.time() - starttime)

        print('Evaluating model...')
        y_pred = pipeline.predict(X_test)
        evaluate_model(y_test, y_pred)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_model.py ../data/processed/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
