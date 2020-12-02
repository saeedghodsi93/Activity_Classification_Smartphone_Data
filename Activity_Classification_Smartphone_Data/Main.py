import os
import csv
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OutputCodeClassifier

# initialization
def init():
	
	print('Init...')
	
	# input and output directories for reading and writing files
	inputdir = 'input files'
	outputdir = 'output files'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)
	
	# choose wether to transform features to zero-mean unit-variance or not
	pre_process = False
	
	# choose wether to use the voting between following classifiers or single classifiers independently. 'single', 'hardvoting', 'softvoting'
	classification_method = 'single'
	
	# choose the set of classifiers. 'knn', 'logistic', 'lda', 'svm', 'tree', 'randomforest', 'extratrees', 'gradboost', 'adaboost', 'mlp', 'ecoc'
	classifier_names = ['svm']
		
	# feature selection method. 'none', 'anova', 'mutualinfo', 'recursivesvm', 'frommodelsvm', 'frommodeltree'
	f_sel_method = 'none'
	
	# parameter tuning search method. 'grid', 'rand'
	tuning_method = 'grid'
	
	return inputdir, outputdir, pre_process, classification_method, classifier_names, f_sel_method, tuning_method
	
# load the dataset
def load_dataset():

	print('Loading Dataset...')
	
	# load the data
	training_features = np.genfromtxt(os.path.join(inputdir, 'X_train.txt'),delimiter=' ')
	training_labels = np.genfromtxt(os.path.join(inputdir, 'y_train.txt'),delimiter=' ')
	test_features = np.genfromtxt(os.path.join(inputdir, 'X_test.txt'),delimiter=' ')
	test_labels = np.genfromtxt(os.path.join(inputdir, 'y_test.txt'),delimiter=' ')
	
	# tolie_features = []
	# tolie_labels = []
	# lieto_features = []
	# lieto_labels = []
	# [tolie_features.append(training_features(idx)) for idx in range(1,len(training_labels)+1) if (training_labels(idx)==9 or training_labels(idx)==11)]
	# [tolie_labels.append(training_labels(idx)) for idx in range(1,len(training_labels)+1) if (training_labels(idx)==9 or training_labels(idx)==11)]
	# [lieto_features.append(training_features(idx)) for idx in range(1,len(training_labels)+1) if (training_labels(idx)==10 or training_labels(idx)==12)]
	# [lieto_labels.append(training_labels(idx)) for idx in range(1,len(training_labels)+1) if (training_labels(idx)==10 or training_labels(idx)==12)]
	
	# make the data zero mean / unit variance
	if pre_process==True:
		scaler = preprocessing.StandardScaler().fit(training_features)
		training_features = scaler.transform(training_features)
		test_features = scaler.transform(test_features)
	
	return training_features,training_labels,test_features,test_labels

# prepare the feature selection and classification pipe, with the set of parameters for the tuning purpose
def single_classifier(clf_name):

	# create the classifier objects
	classifiers = {
		'knn':KNeighborsClassifier(),
		'logistic':LogisticRegression(),
		'lda':LinearDiscriminantAnalysis(),
		'svm':SVC(),
		'tree':DecisionTreeClassifier(),
		'randomforest':RandomForestClassifier(),
		'extratrees':ExtraTreesClassifier(),
		'gradboost':GradientBoostingClassifier(),
		'adaboost':AdaBoostClassifier(),
		'mlp':MLPClassifier(),
		'ecoc':OutputCodeClassifier(SVC(C=2,kernel='linear',shrinking=True,class_weight='balanced'), code_size=2)}

	# feature selection using a pipeline
	if f_sel_method=='none':
		pipe = Pipeline([('clf',classifiers[clf_name])])
		param_set = {}
	elif f_sel_method=='anova':
		pipe = Pipeline([('f_sel',SelectPercentile(score_func=f_classif)), ('clf',classifiers[clf_name])])
		param_set = {'f_sel__percentile':[25,50,75,100]}
	elif f_sel_method=='mutualinfo':
		pipe = Pipeline([('f_sel',SelectPercentile(score_func=mutual_info_classif)), ('clf',classifiers[clf_name])])
		param_set = {'f_sel__percentile':[25,50,75,100]}
	elif f_sel_method=='recursivesvm':
		f_sel = SVC(C=1, kernel='linear', shrinking=True, class_weight='balanced')
		pipe = Pipeline([('f_sel',RFECV(estimator=f_sel)), ('clf',classifiers[clf_name])])
		param_set = {'f_sel__step':[10], 'f_sel__cv':[2], 'f_sel__scoring':['accuracy']}
	elif f_sel_method=='frommodelsvm':
		f_sel = SVC(C=1, kernel='linear', shrinking=True, class_weight='balanced')
		pipe = Pipeline([('f_sel',SelectFromModel(f_sel)), ('clf',classifiers[clf_name])])
		param_set = {}
	elif f_sel_method=='frommodeltree':
		f_sel = ExtraTreesClassifier(n_estimators=100, class_weight='balanced')
		pipe = Pipeline([('f_sel',SelectFromModel(f_sel)), ('clf',classifiers[clf_name])])
		param_set = {}

	# specify parameters of the classifiers
	if clf_name=='knn': #89.9,90.8 'n_neighbors':17, 'p':1, 'weights':'distance'
		param_set.update({'clf__n_neighbors':[1,9,13,17,25,50], 'clf__p':[1,2,3,5], 'clf__weights':['distance'], 'clf__algorithm':['auto'], 'clf__n_jobs':[3]})
	elif clf_name=='logistic': #94.4 'C':1, 'solver':'newton-cg'
		param_set.update({'clf__C':[1,2,3,4], 'clf__solver':['newton-cg'], 'clf__class_weight':['balanced'], 'clf__max_iter':[100]})
	elif clf_name=='lda': #94.9 'solver':'lsqr'
		param_set.update({'clf__solver':['lsqr','eigen'], 'clf__shrinkage':['auto']})
	elif clf_name=='svm': #95.3 'C':1, 'kernel':'linear'
		param_set.update({'clf__C':[0.75,1,1.25,1.5,2], 'clf__kernel':['linear'], 'clf__shrinking':[True], 'clf__probability':[False], 'clf__class_weight':['balanced'], 'clf__decision_function_shape':['ovr']})
	elif clf_name=='tree': #82.3 'max_depth':15
		param_set.update({'clf__min_samples_leaf':[10,50,75,100], 'clf__class_weight':['balanced'], 'clf__presort':[True]})
	elif clf_name=='randomforest': #91.8 'n_estimators':300, 'min_samples_leaf':None, 'max_depth':25
		param_set.update({'clf__n_estimators':[500,1000], 'clf__max_features':[5,10,25], 'clf__min_samples_leaf':[1,10,25] ,'clf__max_depth':[None], 'clf__bootstrap':[True], 'clf__class_weight':['balanced'], 'clf__oob_score':[False], 'clf__n_jobs':[3]})
	elif clf_name=='extratrees': #92.8 'n_estimators':500, 'max_depth':50
		param_set.update({'clf__n_estimators':[100,500,1000], 'clf__max_features':[5,10,20,25,50,100,150], 'clf__min_samples_leaf':[1,10,25,50,100], 'clf__max_depth':[None], 'clf__bootstrap':[False], 'clf__class_weight':['balanced'], 'clf__oob_score':[False], 'clf__n_jobs':[3]})
	elif clf_name=='gradboost': #92.3 'n_estimators':100, 'learning_rate':0.1, 'min_samples_leaf':50
		param_set.update({'clf__n_estimators':[100], 'clf__max_features':['auto'], 'clf__learning_rate':[0.1], 'clf__min_samples_leaf':[50]})
	elif clf_name=='adaboost': #57.9 'n_estimators':100, 'learning_rate':0.1
		param_set.update({'clf__n_estimators':[100,500], 'clf__learning_rate':[0.01,0.1]})
	elif clf_name=='mlp': #95.0 'hidden_layer_sizes':(50,), 'alpha':10, 'solver':'lbfgs'
		param_set.update({'clf__hidden_layer_sizes':[(50,),(60,),(100,)], 'clf__alpha':[0.5,1,2,5,7], 'clf__solver':['adam']})
	elif clf_name=='ecoc':
		param_set.update({})
		
	# run grid search or randomized search
	if tuning_method=='grid':
		search = GridSearchCV(pipe, param_grid=param_set, cv=2, n_jobs=3)
	elif tuning_method=='rand':
		search = RandomizedSearchCV(pipe, param_distributions=param_set, n_iter=10, cv=2, n_jobs=3)
					
	return search

# prepare the voting classifier
def voting_classifier():

	# create the classifier objects
	f_sel = SVC(C=1, kernel='linear', shrinking=True, class_weight='balanced')
	classifiers = {
		'knn':Pipeline([('f_sel',SelectFromModel(f_sel)), ('clf',KNeighborsClassifier())]),
		'logistic':LogisticRegression(),
		'lda':LinearDiscriminantAnalysis(),
		'svm':Pipeline([('f_sel',SelectFromModel(f_sel)), ('clf',SVC())]),
		'tree':DecisionTreeClassifier(),
		'randomforest':RandomForestClassifier(),
		'extratrees':ExtraTreesClassifier(),
		'gradboost':GradientBoostingClassifier(),
		'adaboost':AdaBoostClassifier(),
		'mlp':MLPClassifier(),
		'ecoc':OutputCodeClassifier(SVC(C=2,kernel='linear',shrinking=True,probability=True,class_weight='balanced'), code_size=2)}
		
	# create ensemble of the classifiers
	clfs = []
	[clfs.append((name,classifiers.get(name))) for name in classifier_names]
	
	# create the voting classifier
	voting_type = classification_method[0:4]
	eclf = VotingClassifier(estimators=clfs, voting=voting_type)
	
	# specify parameters of the classifiers
	param_set = {}
	if 'knn' in classifier_names: #89.9,90.8 'n_neighbors':17, 'p':1, 'weights':'distance'
		param_set.update({'knn__clf__n_neighbors':[17], 'knn__clf__p':[1], 'knn__clf__weights':['distance'], 'knn__clf__algorithm':['auto'], 'knn__clf__n_jobs':[3]})
	if 'logistic' in classifier_names: #94.4 'C':1, 'solver':'newton-cg'
		param_set.update({'logistic__C':[2], 'logistic__solver':['lbfgs'], 'logistic__class_weight':['balanced'], 'logistic__max_iter':[100]})
	if 'lda' in classifier_names: #94.9 'solver':'lsqr'
		param_set.update({'lda__solver':['lsqr'], 'lda__shrinkage':['auto']})
	if 'svm' in classifier_names: #95.3 'C':1, 'kernel':'linear'
		param_set.update({'svm__clf__C':[2], 'svm__clf__kernel':['linear'], 'svm__clf__shrinking':[True], 'svm__clf__probability':[True], 'svm__clf__class_weight':['balanced'], 'svm__clf__decision_function_shape':['ovo']})
	if 'tree' in classifier_names: #82.3 'max_depth':15
		param_set.update({'tree__max_depth':[10,15,20], 'tree__class_weight':['balanced'], 'tree__presort':[True]})
	if 'randomforest' in classifier_names: #91.8 'n_estimators':300, 'min_samples_leaf':None, 'max_depth':25
		param_set.update({'randomforest__n_estimators':[100], 'randomforest__max_features':[10,25,50], 'randomforest__min_samples_leaf':[50] ,'randomforest__max_depth':[None], 'randomforest__bootstrap':[True], 'randomforest__class_weight':['balanced'], 'randomforest__oob_score':[True], 'randomforest__n_jobs':[3]})
	if 'extratrees' in classifier_names: #92.8 'n_estimators':500, 'max_depth':50
		param_set.update({'extratrees__n_estimators':[300], 'extratrees__max_features':['auto'], 'extratrees__min_samples_leaf':[50], 'extratrees__max_depth':[None], 'extratrees__bootstrap':[False], 'extratrees__class_weight':['balanced'], 'extratrees__oob_score':[False], 'extratrees__n_jobs':[3]})
	if 'gradboost' in classifier_names: #92.3 'n_estimators':100, 'learning_rate':0.1, 'min_samples_leaf':50
		param_set.update({'gradboost__n_estimators':[100], 'gradboost__max_features':['auto'], 'gradboost__learning_rate':[0.1], 'gradboost__min_samples_leaf':[50]})
	if 'adaboost' in classifier_names:
		param_set.update({'adaboost__n_estimators':[100], 'adaboost__learning_rate':[0.1]})
	if 'mlp' in classifier_names: # 95.0 'hidden_layer_sizes':(50,), 'alpha':10, 'solver':'lbfgs'
		param_set.update({'mlp__hidden_layer_sizes':[(50,)], 'mlp__alpha':[10], 'mlp__solver':['lbfgs']})
	
	# run grid search or randomized search
	if tuning_method=='grid':
		search = GridSearchCV(eclf, param_grid=param_set, cv=2, n_jobs=3)
	elif tuning_method=='rand':
		search = RandomizedSearchCV(eclf, param_distributions=param_set, n_iter=10, cv=2, n_jobs=3)
	
	return search

# utility function to report best score of tuning
def report(clf_name, results, n_top=1):

	candidates = np.flatnonzero(results['rank_test_score'] == 1)
	for candidate in candidates:
		print('\nClassifier: ', clf_name)
		print('Best parameters: {0}'.format(results['params'][candidate]))
		print('Best validation accuracy: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))

# plot the confusion matrix
def plot_confusion_matrix(test_lbls, predicted_lbls):

	# calculate and normalize confusion matrix
	cnf_matrix = confusion_matrix(test_lbls, predicted_lbls)
	cnf_matrix = cnf_matrix.astype('int')
	norm_cnf_matrix = np.copy(cnf_matrix)
	norm_cnf_matrix = norm_cnf_matrix.astype('float')
	for row in range(0,cnf_matrix.shape[0]):
		s = cnf_matrix[row,:].sum()
		if s > 0:
			for col in range(0,cnf_matrix.shape[0]):
				norm_cnf_matrix[row,col] = np.double(cnf_matrix[row,col]) / s
	
	# print confusion matrix
	np.set_printoptions(precision=2)
	print('\nConfusion Matrix=\n', cnf_matrix, '\n', '\nNormalized Confusion Matrix=\n', norm_cnf_matrix, '\n')
	
	# save confusion matrix as text
	np.savetxt(os.path.join(outputdir, 'Confusion Matrix.txt'), cnf_matrix, delimiter='\t', fmt='%d')
	
	# class names
	classes = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying', 'Stand to Sit', 'Sit to Stand', 'Sit to Lie', 'Lie to Sit', 'Stand to Lie', 'Lie to Stand']
	
	# plot confusion matrix
	plt.figure()
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, int(cnf_matrix[i, j]), horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Confusion Matrix.jpg'), bbox_inches='tight', dpi=300)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()
	
	# plot normalized confusion matrix
	plt.figure()
	plt.imshow(norm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = norm_cnf_matrix.max() / 2.
	for i, j in itertools.product(range(norm_cnf_matrix.shape[0]), range(norm_cnf_matrix.shape[1])):
		plt.text(j, i, float("{0:.2f}".format(norm_cnf_matrix[i, j])), horizontalalignment="center", color="white" if norm_cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Normalized Confusion Matrix.jpg'), bbox_inches='tight', dpi=600)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()
	
	return

# calaculate the classification result
def result(predicted_labels):

	print('Calculating Result...')
	
	# print the recognition accuracy
	print('Test accuracy: {0:.3f}'.format(metrics.accuracy_score(test_labels,predicted_labels)))
	
	# plot confusion matrix
	plot_confusion_matrix(test_labels, predicted_labels)
	
	return
	
# classification with parameter tuning
def classification():
	
	print('Classification...')
	
	# test single classifiers
	if classification_method=='single':
		for clf_name in classifier_names:
			search = single_classifier(clf_name)
			search.fit(training_features,training_labels)
			report(clf_name,search.cv_results_)
			lbl = search.predict(test_features)
			print('Test accuracy: {0:.3f}'.format(metrics.accuracy_score(test_labels,lbl)))
	
	# test voting classifer
	elif classification_method=='hardvoting' or classification_method=='softvoting':
		search = voting_classifier()
		search.fit(training_features,training_labels)
		report(classification_method,search.cv_results_)
		lbl = search.predict(test_features)
		result(lbl)
		
	return
	
# program main
if __name__ == '__main__':
	
	inputdir, outputdir, pre_process, classification_method, classifier_names, f_sel_method, tuning_method = init()
	
	training_features,training_labels,test_features,test_labels = load_dataset()
	
	classification()
	