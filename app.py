from flask import Flask
from flask import request
from datetime import datetime
from crypto.caesar import Ceaser, CeaserData
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return "Hello, Denis. Hello Sasha. Hello Kirill. It's me, Konnan'O Brain"

@app.route('/caesar')
def ceaser():
    text = request.args.get('text')
    move = request.args.get('move')
    algorith = Ceaser()
    algorith.ceaser(text, int(move))
    return (algorith.info.toJSON())

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

