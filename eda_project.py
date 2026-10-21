"""EDA and Visualization Assignment II - Akhilesh S (23AM005).
Dataset: Wisconsin Diagnostic Breast Cancer (scikit-learn bundled copy).
Run: python eda_project.py
Outputs: outputs/*.png and outputs/metrics.txt
Educational use only; not for clinical decisions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, roc_curve, precision_recall_fscore_support)
from sklearn.inspection import permutation_importance

SEED = 42
OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # sklearn: 0=malignant, 1=benign
df = X.copy(); df['diagnosis'] = y.map({0:'malignant', 1:'benign'})
# Dataset checks
summary = {'rows': int(len(df)), 'features': int(X.shape[1]), 'missing_cells': int(X.isna().sum().sum()),
           'duplicate_rows': int(df.duplicated().sum()), 'class_counts': {str(k):int(v) for k,v in df.diagnosis.value_counts().items()}}
# Figures: EDA
fig, ax = plt.subplots(figsize=(7,4)); df.diagnosis.value_counts().reindex(['benign','malignant']).plot(kind='bar', ax=ax); ax.set_title('Diagnosis class distribution'); ax.set_xlabel('Diagnosis'); ax.set_ylabel('Observations'); fig.tight_layout(); fig.savefig(OUT/'01_class_distribution.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,5)); X[['mean radius','mean texture','mean perimeter','mean area']].hist(bins=20, ax=ax) if False else None
fig, axes = plt.subplots(2,2,figsize=(9,6));
for col, ax in zip(['mean radius','mean texture','mean perimeter','mean area'], axes.flat): ax.hist(X[col], bins=22, edgecolor='black'); ax.set_title(col); ax.set_xlabel('Value'); ax.set_ylabel('Count')
fig.tight_layout(); fig.savefig(OUT/'02_feature_distributions.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,4)); sns.boxplot(data=df, x='diagnosis', y='mean radius', order=['benign','malignant'], ax=ax); ax.set_title('Mean radius by diagnosis'); fig.tight_layout(); fig.savefig(OUT/'03_radius_boxplot.png', dpi=160); plt.close(fig)
cols=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean concavity','worst radius','worst perimeter']
fig, ax=plt.subplots(figsize=(8,6)); sns.heatmap(X[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax); ax.set_title('Correlation heatmap (selected features)'); fig.tight_layout(); fig.savefig(OUT/'04_correlation_heatmap.png', dpi=160); plt.close(fig)
fig, ax=plt.subplots(figsize=(7,5)); sns.scatterplot(data=df,x='mean radius',y='mean texture',hue='diagnosis',alpha=.8,ax=ax); ax.set_title('Mean radius versus mean texture'); fig.tight_layout(); fig.savefig(OUT/'05_scatter_radius_texture.png',dpi=160); plt.close(fig)
# Stratified 70/15/15 split: test held out, validation used for tuning
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=.50,random_state=SEED,stratify=y_temp)
# Compare candidate architectures/regularization on validation only
candidates=[]
for layers, alpha in [((32,),.001), ((64,32),.001), ((128,64),.001), ((64,32),.01)]:
    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('mlp',MLPClassifier(hidden_layer_sizes=layers,activation='relu',solver='adam',alpha=alpha,learning_rate_init=.001,max_iter=400,early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=SEED))])
    pipe.fit(X_train,y_train); p=pipe.predict(X_val); prob=pipe.predict_proba(X_val)[:,1]
    candidates.append({'layers':str(layers),'alpha':alpha,'validation_accuracy':float(accuracy_score(y_val,p)),'validation_auc':float(roc_auc_score(y_val,prob))})
# Select by validation AUC, tie-break by accuracy
best=max(candidates,key=lambda c:(c['validation_auc'],c['validation_accuracy']))
layers=eval(best['layers'])
# Refit selected architecture on train+validation, preserve test for one final evaluation
X_fit=pd.concat([X_train,X_val]); y_fit=pd.concat([y_train,y_val])
model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('mlp',MLPClassifier(hidden_layer_sizes=layers,activation='relu',solver='adam',alpha=best['alpha'],learning_rate_init=.001,max_iter=400,early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=SEED))])
model.fit(X_fit,y_fit); pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
acc=float(accuracy_score(y_test,pred)); auc=float(roc_auc_score(y_test,prob)); cm=confusion_matrix(y_test,pred,labels=[0,1])
# Model plots
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(model.named_steps['mlp'].loss_curve_); ax.set_title('MLP training loss'); ax.set_xlabel('Iteration'); ax.set_ylabel('Loss'); fig.tight_layout(); fig.savefig(OUT/'06_loss_curve.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['malignant','benign']).plot(ax=ax,cmap='Blues',colorbar=False); ax.set_title('Confusion matrix (test set)'); fig.tight_layout(); fig.savefig(OUT/'07_confusion_matrix.png',dpi=160); plt.close(fig)
fpr,tpr,_=roc_curve(y_test,prob); fig,ax=plt.subplots(figsize=(6,5)); ax.plot(fpr,tpr,label=f'AUC = {auc:.3f}'); ax.plot([0,1],[0,1],'--',label='Chance'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title('ROC curve (positive class: benign)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'08_roc_curve.png',dpi=160); plt.close(fig)
pi=permutation_importance(model,X_test,y_test,n_repeats=10,random_state=SEED,scoring='accuracy')
imp=pd.Series(pi.importances_mean,index=X.columns).sort_values(ascending=False).head(10).sort_values()
fig,ax=plt.subplots(figsize=(8,5)); imp.plot(kind='barh',ax=ax); ax.set_title('Top 10 permutation feature importances'); ax.set_xlabel('Mean decrease in accuracy after shuffling'); fig.tight_layout(); fig.savefig(OUT/'09_feature_importance.png',dpi=160); plt.close(fig)
report=classification_report(y_test,pred,target_names=['malignant','benign'],output_dict=True,zero_division=0)
metrics={'dataset':summary,'split_counts':{'train':len(X_train),'validation':len(X_val),'test':len(X_test),'final_fit_train_plus_validation':len(X_fit)},'tuning_candidates':candidates,'selected_model':{'hidden_layer_sizes':list(layers),'alpha':best['alpha'],'activation':'relu','solver':'adam','learning_rate_init':.001,'max_iter':400,'early_stopping':True,'random_state':SEED},'test_accuracy':acc,'test_roc_auc':auc,'confusion_matrix_labels_order':['malignant','benign'],'confusion_matrix':cm.tolist(),'classification_report':report,'top_permutation_importance':{k:float(v) for k,v in imp.sort_values(ascending=False).items()}}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'metrics.txt').write_text(f"Rows: {len(df)}; features: {X.shape[1]}; missing cells: {summary['missing_cells']}; duplicate rows: {summary['duplicate_rows']}\nSplit train/validation/test: {len(X_train)}/{len(X_val)}/{len(X_test)}\nSelected architecture: {layers}; alpha={best['alpha']}\nTest accuracy: {acc:.4f}\nTest ROC-AUC (positive class benign): {auc:.4f}\nConfusion matrix (rows actual, cols predicted; malignant, benign):\n{cm}\n\nTuning candidates:\n"+'\n'.join(str(c) for c in candidates)+'\n\nClassification report:\n'+classification_report(y_test,pred,target_names=['malignant','benign']))
df.to_csv('breast_cancer_dataset.csv',index=False)
print((OUT/'metrics.txt').read_text())
