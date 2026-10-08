from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")       # 서울 지도


@app.route("/district")
@app.route("/district.html")
def district():
    return render_template("district.html")    # 자치구 페이지 (?name=강남구)


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
