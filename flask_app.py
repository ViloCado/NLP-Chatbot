from flask import Flask, jsonify, render_template_string, request
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Define the function to answer questions
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])
    return answer.replace(' ##', '')

# Create the Flask app
app = Flask(__name__)

# Define the Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_question = request.form['question']
        context = "The symptoms of flu include fever, cough, sore throat, and muscle aches. Common cold can be treated with rest, hydration, and over-the-counter medications. The recommended dosage for ibuprofen is 200-400 mg every 4-6 hours. Side effects of aspirin include stomach pain, heartburn, and gastrointestinal bleeding. Symptoms of COVID-19 include fever, cough, shortness of breath, and loss of taste or smell."
        answer = answer_question(user_question, context)
        return render_template_string(template, question=user_question, answer=answer)
    return render_template_string(template)

@app.route('/api_endpoint', methods=['POST'])
def api_endpoint():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context')
    
    # Replace the following line with your logic to generate an answer
    answer = answer_question(question, context)
    
    return jsonify({'answer': answer})


# HTML template as a string
template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Medical Chatbot</title>
</head>
<body>
    <h1>Medical Chatbot</h1>
    <form method="POST">
        <label for="question">Ask a question:</label><br>
        <input type="text" id="question" name="question"><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if question %}
        <h2>Question:</h2>
        <p>{{ question }}</p>
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
