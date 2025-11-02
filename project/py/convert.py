# testing/convert.py
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Загружаем новую модель
model = joblib.load('./project/py/best_model_v4.sav')

# Явно задаём размерность: 12 фич строго в порядке из features.get_feature_vector
initial_types = [('input', FloatTensorType([None, 12]))]

onnx_model = convert_sklearn(model, initial_types=initial_types)
with open('./testing/model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
print("Saved ONNX to testing/model.onnx")
