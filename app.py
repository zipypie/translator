from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_text as tf_text
import os


# Initialize Flask app
app = Flask(__name__)

# Load both models
model_tagalog_to_cuyonon = tf.saved_model.load('/app/tagalog-cuyonon')
model_cuyonon_to_tagalog = tf.saved_model.load('/app/cuyonon-tagalog')


# Preprocess the input text (normalization, special tokens, etc.)
def preprocess_input(text):
    text = tf.strings.lower(text)  # Convert to lowercase
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')  # Add special tokens
    return text

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    source_lang = data.get("source_lang", "").strip()
    target_lang = data.get("target_lang", "").strip()
    sentence = data.get("sentence", "").strip()

    # Validate input
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400
    if source_lang == target_lang:
        return jsonify({"error": "Source and target languages cannot be the same"}), 400

    # Select the appropriate model based on the source and target languages
    if source_lang == "Tagalog" and target_lang == "Cuyonon":
        model = model_tagalog_to_cuyonon
    elif source_lang == "Cuyonon" and target_lang == "Tagalog":
        model = model_cuyonon_to_tagalog
    else:
        return jsonify({"error": "Invalid language pair"}), 400

    # Preprocess and translate
    input_text = preprocess_input(sentence)
    inputs = tf.convert_to_tensor([input_text])
    translator = model.signatures['serving_default']
    translated_tokens = translator(inputs=inputs)
    translated_text = translated_tokens['output_0'].numpy()[0].decode('utf-8')

    return jsonify({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "original_sentence": sentence,
        "translated_sentence": translated_text
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)


