from PyQt5.QtWidgets import QApplication, QTextEdit, QPushButton, QVBoxLayout, QWidget
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import torch

# Initialize Hugging Face Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Function to Generate Text
def generate_text():
    input_text = user_input.toPlainText()
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_output.setPlainText(output_text)

app = QApplication([])

window = QWidget()
window.setWindowTitle("CatGPT X.X.X V0.1.1 8.4.20XX BUILD betas Client")
window.resize(800, 600)

user_input = QTextEdit()
model_output = QTextEdit()

generate_button = QPushButton('Generate')
generate_button.clicked.connect(generate_text)

layout = QVBoxLayout()
layout.addWidget(user_input)
layout.addWidget(generate_button)
layout.addWidget(model_output)

window.setLayout(layout)
window.show()

sys.exit(app.exec_())
