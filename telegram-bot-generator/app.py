from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sqlite3

app = Flask(__name__)

# Загрузка модели и токенизатора
model_name = './model/custom_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Подключение к базе данных
conn = sqlite3.connect('queries.db', check_same_thread=False)
cursor = conn.cursor()

# Создание таблицы, если она не существует
cursor.execute('''
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY,
    prompt TEXT,
    response TEXT
)
''')

@app.route('/generate', methods=['POST'])
def generate_code():
    prompt = request.json['prompt']
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Сохранение запроса и ответа в базу данных
    cursor.execute('INSERT INTO queries (prompt, response) VALUES (?, ?)', (prompt, generated_code))
    conn.commit()
    
    return jsonify({'generated_code': generated_code})

@app.route('/history', methods=['GET'])
def get_history():
    cursor.execute('SELECT * FROM queries')
    history = cursor.fetchall()
    return jsonify(history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)