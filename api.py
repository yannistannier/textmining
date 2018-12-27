from flask import Flask
from flask import request
from flask_cors import CORS
import json
import spacy
from flask import jsonify
app = Flask(__name__)
CORS(app)



nlp = spacy.load("models/fs_normal/epoch_24")

@app.route('/ner', methods=['POST'])
def hello_world():
	#print(request.json["text"])
	doc = nlp(str(request.json["text"]))
	gene = 0
	disease = 0
	for t in doc:
		if t.ent_type_ == "GENE":
			gene += 1
		if t.ent_type_ == "DISEASE":
			disease += 1
	return jsonify(text=[(t.text, t.ent_type_ if t.ent_type_ else "default") for t in doc], count={"gene":gene, "disease":disease})



@app.route('/austim/<int:id_a>', methods=['GET'])
def autism():
	with open("test/sample_autism.json") as f:
		ok = json.load(t)
    	article = json.loads(ok[id_a])
