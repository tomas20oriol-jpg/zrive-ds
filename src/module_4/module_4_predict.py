from .module_4_fit import DEFAULT_MODEL_FOLDER_PATH
from .push_model import PushModel
from .utils import build_feature_frame
from datetime import datetime
from typing import Any, Dict, Optional

import json
import joblib
import pandas as pd

def load_data(input_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(input_data, orient="index")
    feature_frame = build_feature_frame(df)
    return feature_frame

def load_model(model_path: Optional[str]) -> PushModel:
    if not model_path:
        today_str = datetime.today().strftime("%Y-%m-%d")
        model_path = f"{DEFAULT_MODEL_FOLDER_PATH}/push_{today_str}.joblib"

    if not model_path.exists():
         raise FileNotFoundError(f"File {model_path} does not exist.")
    
    clf = joblib.load(model_path)
    return clf

def handler_predict(event: Dict, _) -> Dict[str, Any]:
        data_to_predict = load_data(event["users"])
        model_path = event.get("model_path", None)
        clf = load_model(model_path)
        predictions = clf.predict(data_to_predict)
        return {
            "statusCode": "200",
            "body": json.dumps({
                "predictions": predictions.to_dict(),
            }),
        }