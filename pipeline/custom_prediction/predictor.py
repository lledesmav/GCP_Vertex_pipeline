import numpy as np
import pandas as pd
from typing import Dict
import pickle

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

class CprPredictor(Predictor):
    
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str): # se va a cargar el 'scaler.pkl' y 'model.pkl' y se van a guardar dentro de las variables self._model y self._scaler
        """Loads the preprocessor and model artifacts."""
        prediction_utils.download_model_artifacts(artifacts_uri)

        model  = pd.read_pickle('model.pkl')
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        self._model = model
        self._scaler = scaler
        
    def preprocess(self, prediction_input: Dict) -> pd.DataFrame:
        instances = prediction_input["instances"] #siempre se va llamar 'instances' cuando haremos uso del endpoint 
        data = pd.DataFrame(instances).astype(float) #primero se pasa como DF
        data = self._scaler.transform(data) # luego se implementa el scaler a toda la 'data'
        return data

    def predict(self, instances: pd.DataFrame) -> Dict:
        """Performs prediction."""
        inputs = instances
        y_hat = self._model.predict(inputs)

        return {"yhat": y_hat.tolist()}
    
    def postprocess(self, prediction_results: Dict) -> Dict:
        return {"predictions": prediction_results["yhat"]}