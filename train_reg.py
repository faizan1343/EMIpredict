import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.models.regression import prep, X_train, y_train
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import joblib

pipe = Pipeline([
    ('prep', prep),
    ('reg', XGBRegressor(
        n_estimators=100, max_depth=5,
        n_jobs=-1, random_state=42, tree_method='hist'))
])
pipe.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(pipe, 'models/reg.pkl')
print('REG saved to models/reg.pkl – DONE')