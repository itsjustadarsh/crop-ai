import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

print("Loading PKL files...")

model = pickle.load(open("model.pkl", "rb"))
minmax = pickle.load(open("minmaxscaler.pkl", "rb"))
standard = pickle.load(open("standscaler.pkl", "rb"))

pipeline = Pipeline([
    ("minmax", minmax),
    ("standard", standard),
    ("model", model)
])

print("Converting to ONNX...")

initial_type = [("float_input", FloatTensorType([None, 7]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    options={id(pipeline): {"zipmap": False}}  # tensor output fix
)

with open("crop_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ crop_model.onnx created!")
