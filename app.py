import os

from flask import Flask, render_template, request, redirect, url_for, flash, session
import datetime

from table import Image

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["FILE_FORMATS"] = ('.png', '.jpg')
app.config["SECRET_KEY"] = os.environ['SECRET_KEY']


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        print(request.files)
        if 'image' in request.files and request.files['image'].filename != '':
            uploaded_file = request.files['image']
            if uploaded_file.filename.endswith(app.config["FILE_FORMATS"]):
                uploaded_filename, fmt = uploaded_file.filename.rsplit(sep=".", maxsplit=1)
                filename = uploaded_filename + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + '.' + fmt
                path = "static/images/"
                full_path = path + filename
                uploaded_file.save(full_path)
                image = Image(full_path)
                result_path = image.get_image()
                session["result_path"] = result_path
                print(session["result_path"])
                return redirect(url_for('index'))
            flash(f"Поддерживаемые форматы изображения: {' '.join(app.config['FILE_FORMATS'])}", 'error')
            return redirect(url_for('index'))
        flash("Отправтье изображение", 'error')
        return redirect(url_for('index'))

    print(session.get('result_path'))
    result_path = session.get('result_path')
    if result_path:
        del session['result_path']
    return render_template("index.html", **{'result_path': result_path})
