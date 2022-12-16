from joblib import dump, load
from xgboost import XGBClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard,ExplainerHub
import pandas as pd
import numpy as np


xgmodel=load("xgbst_model")
X_val=pd.read_feather('X_val.fthr')
y_val=pd.read_feather('y_val.fthr')


model = load("model.joblib")
X = pd.read_parquet("X_test.parquet.gzip")
Y = pd.read_parquet('y_test.parquet.gzip')



total_Sample=X_val.shape[0]
sample_count=10000
sample_index=np.random.randint(0,total_Sample,sample_count)
a=X.shape[0]
a_count=10000
a_index=np.random.randint(0,a,a_count)
explaine2 = ClassifierExplainer(model, X.iloc[a_index], Y.iloc[a_index])



explainer = ClassifierExplainer(xgmodel, X_val.iloc[sample_index], y_val.iloc[sample_index])

db1 = ExplainerDashboard(explainer, title="Digital Churn", name="db1",
            description="This is model option one")
db2 = ExplainerDashboard(explaine2, title="Loan Pricing Model", name="db2",
            description="This is model option one")
hub = ExplainerHub([db1,db2], title="ABB ML Explainer",
            description="Tool to deep dive into mechanics of ML Models including feature importance")

hub.run(port=9294)