import pandas as pd
import numpy as np

print(np.__version__)
print(pd.__version__)

d = pd.read_csv("data.csv")

bmws = d[d["Make"] == "BMW"]
print(bmws.MSRP.mean())

print(len(d[d.Year >= 2015]["Engine HP"]) - d[d.Year >= 2015]["Engine HP"].count())

EngineHPs = d[d.Year >= 2015]["Engine HP"]
mean = EngineHPs.mean()
filled = EngineHPs.fillna(mean)
print(round(mean))
print(round(filled.mean()))

rr = d[d["Make"] == "Rolls-Royce"]
small = rr[["Engine HP", "Engine Cylinders", "highway MPG"]]
deduped = small.drop_duplicates()

X = np.array(deduped)
XTX = np.dot(X.T, X)

print(np.linalg.inv(XTX).sum())

y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])

print(np.dot(np.dot(np.linalg.inv(XTX), X.T), y)[0])
