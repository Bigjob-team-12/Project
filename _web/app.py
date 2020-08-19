from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_admin.contrib.fileadmin import FileAdmin
from datetime import datetime
#from werkzeug import secure_filename 보안 수정 예정
import os

basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(basedir, 'images/uploads')

##################################################################
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.sqlite"
app.config['SECRET_KEY'] = 'cat'
app.config['UPLOADED_PATH'] = upload_dir
##################################################################
db = SQLAlchemy(app)
admin = Admin(name='찾아주CatDog - Admin')
admin.init_app(app)
##################################################################

class QueryLostAnimals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    querydate = db.Column(db.String, unique=True, nullable=False)
    location = db.Column(db.String, nullable=False)
    lostdate = db.Column(db.String, nullable=False)
    animal = db.Column(db.String, nullable=False)
    filename = db.Column(db.String, unique=True, nullable=False)

##################################################################

admin.add_view(ModelView(QueryLostAnimals, db.session, name='요청 목록'))
admin.add_view(FileAdmin(upload_dir, name='이미지 파일 관리'))

##################################################################

@app.route('/')
def mainpage():
    return render_template('index.html')

@app.route('/find1', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        time = str(datetime.now())
        f.save(os.path.join(app.config['UPLOADED_PATH'],
                            time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + f.filename))
        db.session.add(QueryLostAnimals(querydate=time, location=request.form['location'], lostdate=request.form['date'], animal=request.form['animal'], filename=time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + f.filename))
        db.session.commit()
        print('done')
    return render_template('find_my_dog_q.html')

##################################################################


if __name__ == '__main__':
    app.run()