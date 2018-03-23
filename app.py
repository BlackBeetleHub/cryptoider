from flask import Flask
from flask import request
from datetime import datetime
from crypto.caesar import Ceaser, CeaserData
from crypto.des import Des
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return "Hello, Denis. Hello Sasha. Hello Kirill."


@app.route('/caesar')
def ceaser():
    text = request.args.get('text')
    move = request.args.get('move')
    algorith = Ceaser()
    algorith.ceaser(text, int(move))
    return (algorith.info_steps.toJSON())


@app.route('/des')
def des():
    type = request.args.get('type')
    m = request.args.get('massage')
    k = request.args.get('key')
    n_rounds = request.args.get('round')
    if len(m) % 8 != 0:
        for x in range(0, 8 - len(m) % 8):
            m += " "
    message, key = Des.str_to_bit_array(m), Des.str_to_bit_array(k)
    if type == 'encrypt':

        ecrypted_m = Des.des_encrypt(message, key, int(n_rounds))

        return ecrypted_m
    else:
        decrypred_m = Des.des_decrypt(message, key, n_rounds)
        return decrypred_m


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
