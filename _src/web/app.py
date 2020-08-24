from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import sqlite3
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

""" 사용자 요청 저장 """
class QueryLostAnimals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    querydate = db.Column(db.String, unique=True, nullable=False)
    location = db.Column(db.String, nullable=False)
    lostdate = db.Column(db.String, nullable=False)
    animal = db.Column(db.String, nullable=False)
    filename = db.Column(db.String, unique=True)

""" 탐색 대상 데이터 (테스트용 임시 DB) """
class TempList(db.Model):
    NO = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String)
    kind = db.Column(db.String)
    color = db.Column(db.String)
    sex = db.Column(db.String)
    neutralization = db.Column(db.String)
    age_weight = db.Column(db.String)
    date = db.Column(db.String)
    location = db.Column(db.String)
    deadline = db.Column(db.String)
    image = db.Column(db.String)
    url = db.Column(db.String)
    time = db.Column(db.String)

##################################################################

admin.add_view(ModelView(QueryLostAnimals, db.session, name='요청 목록'))
admin.add_view(FileAdmin(upload_dir, name='이미지 파일 관리'))

##################################################################

@app.route('/')
def mainpage():
    return render_template('index.html')

""" 사용자 요청 페이지 """
@app.route('/find_my_q', methods=['GET', 'POST'])
def ask():
    # 사용자 요청 발생 시
    if request.method == 'POST':
        f = request.files['file']

        # metadata 저장
        time = str(datetime.now())
        qla = QueryLostAnimals(querydate=time, location=request.form['location'], lostdate=request.form['date'], animal=request.form['animal'])
        db.session.add(qla)
        db.session.commit()

        # 파일 저장
        fn = (time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + str(qla.id)+'.'+f.filename.split('.')[-1])
        f.save(os.path.join(app.config['UPLOADED_PATH'], fn))
        qla.filename = fn
        db.session.commit()

        # 결과 페이지로 이동
        return redirect('/find_my_a?id='+str(qla.id))

    # 기본 화면
    return render_template('find_my_dog_q.html')

""" 결과 페이지 """
@app.route('/find_my_a', methods=['GET', 'POST'])
def answer():
    # get: 사용자 요청 고유번호(id) 및 페이지 번호(page)
    if request.method == 'GET':
        page = request.args.get('page', type=int, default=1)

        # 사용자 요청 이미지 노출; 로컬 로딩 문제 해결중
        asked = list(dbquery('query_lost_animals', request.args.get('id'))[0])
        asked[5] = 'images/uploads/'+asked[5]

        # 유사도 높은 이미지 DB에서 로드
        sims = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        found = dbquery('temp_list', sims)
        return render_template('find_my_dog_a.html', id=request.args.get('id'), page=page, asked=asked, found=found)


""" DB 탐색 query """
def dbquery(db, id, start=0, end=0):
    conn = sqlite3.connect(os.path.join(basedir, 'database.sqlite'))
    cur = conn.cursor()
    if db == 'query_lost_animals':
        cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))
    elif db == 'temp_list':
        cur.execute('SELECT * FROM {} WHERE NO in {}'.format(db, id))
    rows = cur.fetchall()
    conn.close()
    return rows
##################################################################


if __name__ == '__main__':
    app.run()