from flask import Flask, render_template, template_rendered

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/ecg')
def about():
    return render_template("ecg.html")


if __name__=="__main__":
    app.run(debug=True)
