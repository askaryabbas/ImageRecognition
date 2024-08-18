from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, "model/mobilenet_v2-b0353104.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "images/flower.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)