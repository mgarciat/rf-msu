from rfmsu import RandomForestMSU
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

rf = RandomForestMSU(
    numIterations=100,
    kValue=0,
    minNum=1.0,
    maxDepth=0,
    bootstrap_mode="weighted",
    rediscretize_msu_per_node=True,
    n_jobs=8,
    debug=False
)

rf.fit(
    X,
    y,
    feature_types=["numeric"] * X.shape[1]
)

predicciones = rf.predict(X)

print(predicciones)
