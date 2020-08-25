from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import pymysql
import pandas as pd


'''
province = {
    '서울': ['서울', '인천', '경기'],
    '인천': ['인천', '서울', '경기'],
    '대전': ['대전', '세종', '충북', '충남'],
    '대구': ['대구', '경북', '경남'],
    '울산': ['울산', '부산', '경북', '경남'],
    '부산': ['부산', '울산', '경남'],
    '광주': ['광주', '전남'],
    '세종': ['세종', '대전', '충북', '충남'],
    '경기': ['서울', '인천', '강원', '충북', '충남'],
    '강원': ['강원', '경기', '충북', '경북'],
    '충북': ['충북', '대전', '세종', '경기', '강원', '충남', '경북', '전북'],
    '충남': ['충남', '대전', '세종', '경기', '충북', '전북'],
    '경북': ['경북', '대구', '울산', '강원', '충북', '경남', '전북'],
    '경남': ['경남', '대구', '울산', '부산', '경북', '전북', '전남'],
    '전북': ['전북', '충북', '충남', '경북', '경남', '전남'],
    '전남': ['전남', '광주', '경남', '전북'],
    '제주': ['제주'] 
    }
'''


basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(basedir, 'static/images/uploads')

##################################################################
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:bigjob12@localhost:3306/project"
app.config['SECRET_KEY'] = 'cat'
app.config['UPLOADED_PATH'] = upload_dir
##################################################################
db = SQLAlchemy(app)
##################################################################

""" 사용자 요청 저장 """
class QueryLostAnimals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    querydate = db.Column(db.String, unique=True, nullable=False)
    location = db.Column(db.String, nullable=False)
    lostdate = db.Column(db.String, nullable=False)
    animal = db.Column(db.String, nullable=False)
    filename = db.Column(db.String, unique=True)

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
        ttime = datetime.now()
        time = str(ttime)
        qla = QueryLostAnimals(querydate=time, location=request.form['location'], lostdate=request.form['date'], animal=request.form['animal'])
        db.session.add(qla)
        db.session.commit()

        # 파일 저장
        fn = (time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + str(qla.id)+'.'+f.filename.split('.')[-1])
        f.save(os.path.join(app.config['UPLOADED_PATH'], fn))
        qla.filename = fn
        db.session.commit()

        # 쿼리 이미지 분류기에 넘기기


        # 작업 소요시간 확인
        print(datetime.now()-ttime)

        # 결과 페이지로 이동
        return redirect('/find_my_a?id='+str(qla.id))

    # 기본 화면
    return render_template('find_my_dog_q.html')

""" 결과 페이지 """
@app.route('/find_my_a', methods=['GET', 'POST'])
def answer():
    ITEMPERPAGE = 10 # 페이지 당 노출 아이템 수
    # get: 사용자 요청 고유번호(id) 및 페이지 번호(page)
    if request.method == 'GET':
        page = request.args.get('page', type=int, default=1)

        # 사용자 요청 이미지 노출; 로컬 로딩 문제 해결중
        asked = list(dbquery('query_lost_animals', request.args.get('id'))[0])
        asked[5] = 'images/uploads/'+asked[5]

        # 유사도 높은 이미지 DB에서 로드
        sims = pd.read_csv('../data_analysis/re_id/test/result.csv', names=['rank', 'number'], header=0)
        found = dbquery('protect_animals_url1', tuple(sims['number'].values))
        found = sims.merge(pd.DataFrame(pd.DataFrame(found,
                                                     columns=['no', 'number', 'kind', 'color', 'sex', 'neutralization',
                                                              'age_weight', 'date', 'location', 'characteristic',
                                                              'deadline', 'center_name', 'center_number',
                                                              'center_address', 'image', 'url', 'time'])))
        return render_template('find_my_dog_a.html', id=request.args.get('id'), page=page, asked=asked, found=found[found.columns[1:]].values.tolist()[(page-1)*ITEMPERPAGE:page*ITEMPERPAGE])


""" DB 탐색 query """
def dbquery(db, id, start=0, end=0):
    conn = pymysql.connect(host='localhost', user='root', password='bigjob12', db='project', charset='utf8')
    cur = conn.cursor()
    if db == 'query_lost_animals':
        cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))
    elif db == 'protect_animals_url1':
        cur.execute('SELECT * FROM {} WHERE number in {}'.format(db, id))
    rows = cur.fetchall()
    conn.close()
    return rows
##################################################################


if __name__ == '__main__':
#    pd.set_option('display.max_rows', None)
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', None)
#    pd.set_option('display.max_colwidth', -1)

    def test():
        print(datetime.now())
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(test, 'interval', minutes=1)
    # interval: 서비스 시작 이후 일정 기간이 지날 때마다 반복 (시작했을 때 수행되지는 않음)
    # cron: 지정된 시간에 수행

    app.run()