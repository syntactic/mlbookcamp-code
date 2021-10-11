import sklearn
import numpy as np
import pickle

dv_pickle = "/Users/syntactic/CodeSandbox/Python/mlbookcamp-code/course-zoomcamp/05-deployment/dv.bin"
model_pickle = "/Users/syntactic/CodeSandbox/Python/mlbookcamp-code/course-zoomcamp/05-deployment/model1.bin"

dv_pickle_file = open(dv_pickle, 'rb')
dv = pickle.load(dv_pickle_file)
dv_pickle_file.close()

model_file = open(model_pickle, 'rb')
model = pickle.load(model_file)
model_file.close()

question_3_customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
question_3_customer_vectorized = dv.transform([question_3_customer])

print(model.predict_proba(question_3_customer_vectorized))
