from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the trained model
class HybridModel(nn.Module):
    def __init__(self, model, mode):
        super(HybridModel, self).__init__()
        self.model = model
        self.mode = mode
        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 * 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.permute(0, 2, 1)
        x = self.cnn(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load trained model
model = HybridModel(bert_model, mode="cnn_lstm")
model.load_state_dict(torch.load("cnn_lstm_model_bert.pth", map_location=device))
model.to(device)
model.eval()

def predict_message(message):
    inputs = tokenizer(
        message, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = int((output > 0.5).cpu().numpy())
    return prediction

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Phishing Detection API is running! Use /predict with POST requests."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    prediction = predict_message(message)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)