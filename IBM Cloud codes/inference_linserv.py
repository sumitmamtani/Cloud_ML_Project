from __future__ import print_function
from flask import Flask, json, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from flask import jsonify
import flask
app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
transform = T.Compose([T.Grayscale(num_output_channels=1),
                      T.Resize(28),
                      T.CenterCrop(28),
                      T.ToTensor(),
                      T.Normalize((0.1307,), (0.3081,))])
def preprocess(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image
def infer(image):
    model = Net()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    output = model(image)
    result = output.squeeze().argmax().item()
    return result

@app.route("/predict", methods=["GET", "POST", "OPTIONS", "HEAD", "PUT"])
def predict():
  try:
    if request.method == "POST":
            print("Recieved a Request")
            print("===>", request.files)
            file = request.files["inputFile"]
            model = Net().to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
            model.load_state_dict(torch.load('model_weights.pth'), strict=False)
            model.eval()
            res = {}
            if not file:
                res['status'] = 'missing image'
            else:
                res['status'] = 'success'
                image = Image.open(file.stream)
                output = infer(preprocess(image))
                
            resp = flask.Response(json.dumps(res))
            resp = flask.Response(json.dumps({
                "label": output,
                # "confidence": str(round(torch.max(percentage, 0)[0].item(), 4)),
                "string": "The predicted class is "+output
            }), mimetype="application/json")
            resp.headers['Access-Control-Allow-Origin'] = "*"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Methods"] = "GET,HEAD,OPTIONS,POST,PUT"
            resp.headers["Access-Control-Allow-Headers"] = "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers"
            return resp
    else:
        resp = flask.Response(jsonify({
        }), mimetype="application/json")
        resp.headers['Access-Control-Allow-Origin'] = "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,HEAD,OPTIONS,POST,PUT"
        resp.headers["Access-Control-Allow-Headers"] = "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers"
        return resp
  except Exception as e:
            print(e)
            resp = flask.Response(jsonify({
                }),
                mimetype="application/json"
            )
            resp.headers['Access-Control-Allow-Origin'] = "*"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Methods"] = "GET,HEAD,OPTIONS,POST,PUT"
            resp.headers["Access-Control-Allow-Headers"] = "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers"
            return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=49101, debug = True)