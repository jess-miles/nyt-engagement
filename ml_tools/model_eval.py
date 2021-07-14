import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import joblib

def eval_clf_model(clf, X_test, y_test, X_train, y_train, score='std',
               reports=True, labels=['Class 0', 'Class 1'], 
               normalize_cm='true'):
    """Shows metrics and plots visualizations to interpret classifier model 
    performance.
    
    ***
    Args
    
    clf: classifier model to evaluate (or fit pipeline with a clf model as
    the last step)
    
    X_test: dataframe of test predictors
    
    y_test: dataframe of true target values
    
    X_train: dataframe (optional). Default is None. Provide training data if
    you want to evaluate performance on train versus test; otherwise only 
    test performance is evaluated.
    
    y_train: dataframe (optional). Default is None. Provide training data if
    you want to evaluate performance on train versus test; otherwise only 
    test performance is evaluated.
    
    score: string (optional). Default is `std` to return standard F1, accuracy, 
    and recall scores. Use `macro` to return macro F1 and recall, and balanced
    accuracy. Scores are always returned, regardles of `reports` param.
    
    reports: boolean (optional). Default is True. Set to False to return only 
    scores, not actual classification reports.
    
    labels: list (optional). Provide a list of labels for the target class.
    
    normalize_cm: string, default `true`. Setting for whether and how to
    normalize the confusion matrix. See sklearn documentation for options.
    """
    multi = True if len(labels) > 2 else False
    
    spacer = '*'*30
    
    # Get predictions 1 time only, since they will be used in a few spots
    test_preds = clf.predict(X_test)
    train_preds = clf.predict(X_train)
    
    #if multi:
    #    test_predict_proba = clf.predict_proba(X_test)
    
    # print classification reports
    if reports:
        print(spacer + ' Training Data ' + spacer)
        print(metrics.classification_report(y_train, train_preds))
        print()
        print(spacer + ' Test Data ' + spacer)
        print(metrics.classification_report(y_test, test_preds))
        print()
    
    # print scores from train and test next to each other for easy comparison
    print(spacer + ' Training Scores ' + spacer)

    # Train
    if score == 'std':
        train_f1 = np.round(metrics.f1_score(y_train, train_preds), 4)
        print(f"                  Training F1 = {train_f1}")
        train_r = np.round(metrics.recall_score(y_train, train_preds), 4)
        print(f"              Training Recall = {train_r}")
        train_acc = np.round(metrics.accuracy_score(y_train, train_preds), 4)
        print(f"            Training Accuracy = {train_acc}")
    elif score == 'macro':
        train_f1m = np.round(metrics.f1_score(y_train, train_preds, average='macro'), 4)
        print(f"            Training Macro F1 = {train_f1m}")
        train_rm = np.round(metrics.recall_score(y_train, train_preds, average='macro'), 4)
        print(f"        Training Macro Recall = {train_rm}")
        train_accbal = np.round(metrics.balanced_accuracy_score(y_train, train_preds), 4)
        print(f"   Training Balanced Accuracy = {train_accbal}")
    print()
    print(spacer + ' Test Scores ' + spacer)
    
    #Test
    if score == 'std':
        test_f1 = np.round(metrics.f1_score(y_test, test_preds), 4)
        print(f"                      Test F1 = {test_f1}")
        test_r = np.round(metrics.recall_score(y_test, test_preds), 4)
        print(f"                  Test Recall = {test_r}")
        test_acc = np.round(metrics.accuracy_score(y_test, test_preds), 4)
        print(f"                Test Accuracy = {test_acc}")
        
    elif score == 'macro':
        test_f1m = np.round(metrics.f1_score(y_test, test_preds, average='macro'), 4)
        print(f"                Test Macro F1 = {test_f1m}")
        test_rm = np.round(metrics.recall_score(y_test, test_preds, average='macro'), 4)
        print(f"            Test Macro Recall = {test_rm}")
        test_accbal = np.round(metrics.balanced_accuracy_score(y_test, test_preds), 4)
        print(f"       Test Balanced Accuracy = {test_accbal}")
    print()
    print(spacer + ' Differences ' + spacer)
    
    #Diffs
    if score == 'std':
        print(f"               Train-Test F1 Diff = {test_f1 - train_f1}")       
        print(f"           Train-Test Recall Diff = {test_r - train_r}")       
        print(f"         Train-Test Accuracy Diff = {test_acc - train_acc}")     
    elif score == 'macro':  
        print(f"         Train-Test Macro F1 Diff = {test_f1m - train_f1m}")      
        print(f"     Train-Test Macro Recall Diff = {test_rm - train_rm}")       
        print(f"Train-Test Balanced Accuracy Diff = {test_accbal - train_accbal}")
    
    print()
    print(spacer + ' Graphs for Test ' + spacer)
    
    # plot graphs
    
    if not multi:
        auc = np.round(metrics.roc_auc_score(y_test, test_preds), 2)
        
        ap = np.round(metrics.average_precision_score(y_test, test_preds), 2)

        fig, [ax1, ax2, ax3] = plt.subplots(figsize=[10, 3], nrows=1, ncols=3)
        plt.tight_layout(pad=2.5)
        metrics.plot_confusion_matrix(clf, X_test, y_test, 
                normalize=normalize_cm, display_labels=labels, 
                                      cmap='Reds', ax=ax1)
        metrics.plot_roc_curve(clf, X_test, y_test, ax=ax2)
        ax2.legend(loc='best', fontsize='small', labels=[f'AUC: {auc}'])

        metrics.plot_precision_recall_curve(clf, X_test, y_test, ax=ax3)
        ax3.legend(loc='best', fontsize='small')
        ax3.legend(loc='best', fontsize='small', labels=[f'AP: {ap}'])
        plt.show();
    
    #if multi-class, just plot confusion matrix
    else:
        fig, ax1 = plt.subplots(figsize=[6, 4])
        plt.tight_layout(pad=2.5)
        metrics.plot_confusion_matrix(clf, X_test, y_test, 
                normalize=normalize_cm, display_labels=labels, cmap='Reds', 
                                        ax=ax1)
        plt.show();
        
    
    return None


def clf_gridsearch_wpipe(clf_pipe, grid_params, X_train, y_train, X_test, y_test,
                     class_labels, file_name, save_path, 
                     scoring='recall', score_type='std', n_jobs=-1, verbose=1,
                     normalize_cm='true'):
    """
    Uses provided `clf_pipe` and `grid_params` to perform a GridSearchCV on best
    params according to specified `scoring` metric. See sklearn documentation 
    for available scoring metrics.
    
    Once best estimator is found, both the best estimator and the entire 
    GridSearchCV object are dumped to file using joblib. `file_name` and 
    `save_path` used in this exporting to file.
    
    normalize_cm: string, default `true`. Setting for whether and how to
    normalize the confusion matrix. See sklearn documentation for options.
    """
    gs = GridSearchCV(clf_pipe, grid_params, n_jobs=n_jobs, verbose=verbose,
                         scoring=scoring)

    # run the gridsearch
    gs.fit(X_train, y_train)

    # print best estimator params and score
    print(gs.best_estimator_)
    print(gs.best_score_)

    # dump out best estimator and gs object to gdrive
    joblib.dump(gs.best_estimator_.named_steps['clf'], 
                f"{save_path}BestEst_{file_name}.joblib.gz")
    print()
    print(f"Saved best estimator to: {save_path}BestEst_{file_name}.joblib.gz")
    joblib.dump(gs, f"{save_path}GSObject_{file_name}.joblib.gz")
    print()
    print(f"Saved GridSearch object to: {save_path}GSObject_{file_name}.joblib.gz")

    # print the classifier model report
    eval_clf_model(gs, X_test, y_test, X_train, y_train, labels=class_labels,
                  normalize_cm=normalize_cm, score=score_type)

    return None


def load_rebuild_eval_bestpipe(gsfile_name, X_train, y_train, X_test, y_test, 
                              class_labels, load_path=''):
    """
    Loads and rebuilds the best pipeline (including estimator) from a 
    GridSearch object that was dumped to file using `joblib`. Fits on
    X_train and y_train, and runs classifier evaluation function to show
    model performance.
    
    Returns new pipeline object built using the best params from the gridsearch,
    and the gridsearch object itself, so it can be queried to show its best
    score.
    
    `load_path` should be the path to the gridsearch file to load in, if not
    in current directory.
    """
    
    gs = joblib.load(load_path + gsfile_name)
    
    pipe = Pipeline(gs.best_estimator_.get_params()['steps'])
    
    print(gs.best_estimator_.get_params()['steps'])
    print()
    print(f"Best score from GS-CV: {np.round(gs.best_score_, 3)}")
    print()

    pipe.fit(X_train, y_train)

    eval_clf_model(pipe, X_test, y_test, X_train, y_train, 
                   labels=class_labels)
    
    return pipe, gs


def custom_confusion_matrix(df, title):
    """Plots a confusion matrix using a dataframe compiled manually.
    """
    with sns.plotting_context(context='poster'):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df, cmap='Reds', ax=ax, annot=True, fmt='.0%', square=False, )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.yticks(rotation=0)
        ax.set_title(title);