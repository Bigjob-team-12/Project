from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import pymysql
import pickle
import os
import re
import sys
import shutil
import call_fun
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

""" login information"""
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
#SMTP_PASSWORD = ""  # 비밀번호 초기 설정
#pickle.dump(SMTP_PASSWORD, open("pw.pickle", 'wb'))
SMTP_USER = "findmycatdog@gmail.com"
SMTP_PASSWORD = pickle.load(open('pw.pickle', 'rb'))

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
        qla = QueryLostAnimals(querydate=time, location=request.form['location'], lostdate=request.form['date'],
                               animal=request.form['animal'])
        db.session.add(qla)
        db.session.commit()

        # 파일 저장
        fn = (time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + str(qla.id) + '.' +
              f.filename.split('.')[-1])
        f.save(os.path.join(app.config['UPLOADED_PATH'], fn))
        qla.filename = fn
        db.session.commit()
        shutil.copy(os.path.join(app.config['UPLOADED_PATH'], fn), os.path.join('./static/images/input_image', fn))
        shutil.copy(os.path.join(app.config['UPLOADED_PATH'], fn), os.path.join('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list', fn))

        # 쿼리 이미지 분류기에 넘기기
        call_fun.main()

        os.remove('./static/images/input_image'+'/'+fn)
        os.remove('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list'+'/'+fn)
        # 작업 소요시간 확인
        print('소요된 시간 :' + str(datetime.now() - ttime))

        # 결과 페이지로 이동
        return redirect('/find_my_a?id=' + str(qla.id))

    # 기본 화면
    return render_template('find_my_dog_q.html')


""" 결과 페이지 """


@app.route('/find_my_a', methods=['GET', 'POST'])
def answer():
    ITEMPERPAGE = 10  # 페이지 당 노출 아이템 수
    # get: 사용자 요청 고유번호(id) 및 페이지 번호(page)
    if request.method == 'GET':
        page = request.args.get('page', type=int, default=1)
        register = request.args.get('register', type=str, default=None)

        # 사용자 요청 이미지 노출; 로컬 로딩 문제 해결중
        asked = list(dbquery(db='query_lost_animals', id=request.args.get('id'))[0])
        asked[5] = 'images/uploads/' + asked[5]

        # 유사도 높은 이미지 DB에서 로드
        sims = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv', names=['rank', 'number'], header=0)
        found = dbquery(db='protect_animals_url1', id=tuple(sims['number'].values))
        found = sims.merge(pd.DataFrame(pd.DataFrame(found,
                                                     columns=['no', 'number', 'kind', 'color', 'sex', 'neutralization',
                                                              'age_weight', 'date', 'location', 'characteristic',
                                                              'deadline', 'center_name', 'center_number',
                                                              'center_address', 'image', 'url', 'time'])))
        return render_template('find_my_dog_a.html', id=request.args.get('id'), page=page, asked=asked,
                               found=found[found.columns[1:]].values.tolist()[
                                     (page - 1) * ITEMPERPAGE:page * ITEMPERPAGE], register=register,
                               path=request.full_path)

    # 사용자 푸시 알림 신청 시
    if request.method == 'POST':
        if re.compile('[a-zA-Z0-9+-_.]+').match(request.form['emailid']) and re.compile(
                '[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$').match(request.form['domain']):
            # 정상적인 이메일이 입력되었을 경우 DB에 저장
            mailaddress = request.form['emailid'] + '@' + request.form['domain']
            query = "INSERT INTO query_need_push VALUES ({}, '{}', 1)".format(request.form['id'], mailaddress)
            dbquery(insert=True, query=query)
            register = 'Yes'
        else:
            # 올바른 이메일이 입력되지 않았을 경우
            register = 'No'
        return redirect('/find_my_a?id=' + request.form['id'] + '&register=' + register)


""" DB query """
def dbquery(db=None, id=None, insert=False, query=None):
    conn = pymysql.connect(host='localhost', user='root', password='bigjob12', db='project', charset='utf8')
    cur = conn.cursor()

    # insert
    if insert == True:
        cur.execute(query)
        conn.commit()
        conn.close()
        return

    # select
    else:
        # 사용자 요청 목록
        if db == 'query_lost_animals':
            cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))
        # 공고 데이터 DB
        elif db == 'protect_animals_url1':
            cur.execute('SELECT * FROM {} WHERE number in {}'.format(db, id))
        # 사용자 push 요청 목록
        elif db == 'query_need_push':
            cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))

        rows = cur.fetchall()
        conn.close()
        return rows

""" e-mail push """
def send_mail(queryid, newfoundid):
    # 사용자 id 및 찾은 공고번호 기반 DB 탐색
    queryuser = list(dbquery(db='query_need_push', id=queryid)[0])
    newfound = list(dbquery(db='protect_animals_url1', id=newfoundid))

    # 푸시 알림을 신청한 사용자가 맞을 경우
    if queryuser[2] == 1:
        # message header
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = queryuser[1]
        msg["Subject"] = '찾아줘 CatDog: 유사한 개체가 탐지되었습니다.'

        # message 본문
        msg.attach(MIMEText(
            '<html><body><h3>등록하신 개체와 유사한 개체가 탐지되었습니다.</h3><table border="1" bordercolor="#F8F8F8" align = "center"><thead bgcolor="#F4F4F4" align ="center"><tr><th rowspan="2"></th><th>공고번호</th><th>발견일자</th><th>발견장소</th><th colspan="2">공고기간</th><th>보호소</th><th>바로가기</th></tr><tr><th>공고 상 분류</th><th>색상</th><th>성별</th><th>중성화 여부</th><th>추정 연령</th><th>몸무게</th><th>특이사항</th></tr></thead><tbody align ="center">',
            'html', 'utf-8'))
        for item in newfound:
            msg.attach(MIMEText(
                '<tr><td rowspan="2"><img src="{}" height=100px></td><td>{}</td><td>{}</td><td>{}</td><td colspan="2">{}</td><td>{}</td><td><a href="{}"><b>공고</b></a></td></tr><tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(item[14], item[1], item[7], item[8], item[10], item[11], item[15], item[2], item[3], item[4], item[5], item[6].split('/')[0][:-1], item[6].split('/')[1][1:], item[9]),
                'html', 'utf-8'))
        msg.attach(MIMEText('</tbody></table></body></html>', 'html', 'utf-8'))

        # 푸시 거부 로직 추가 필요

        # send
        with smtplib.SMTP_SSL(SMTP_SERVER) as s:
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(SMTP_USER, msg["To"], msg.as_string())

##################################################################


if __name__ == '__main__':
#    pd.set_option('display.max_rows', None)
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', None)
#    pd.set_option('display.max_colwidth', -1)

    app.run()
    #send_mail(72, ('경기-부천-2020-00733', '경북-경주-2020-00736'))

"""


def test():
    print(datetime.now())


scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(test, 'interval', minutes=1)

app.run()
"""
