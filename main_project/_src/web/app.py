from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_dropzone import Dropzone
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
import math
from numba import cuda
from werkzeug.datastructures import ImmutableMultiDict
import tensorflow as tf
from tensorflow import keras
import re_run
# path 설정
# sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/dog_image_similarity')
# import predict_dog_data
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/dog_image_similarity')
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code')
import extract_similar_image_path, copy_image, predict_dog_data
import reid_query


model = predict_dog_data.make_model(256)
sys.path.pop()



# print(sys.path.pop())
#
# print(model)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = tf.compat.v1.ConfigProto()
# config.gpu_option.allow_growth = True
basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(basedir, 'static/images/uploads')

##################################################################

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:bigjob12@localhost:3306/project"
app.config['SECRET_KEY'] = 'cat'
app.config['UPLOADED_PATH'] = upload_dir
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
##################################################################
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'
app.config['DROPZONE_MAX_FILES'] = 1
app.config['DROPZONE_IN_FORM'] = True
app.config['DROPZONE_UPLOAD_ON_CLICK'] = True
app.config['DROPZONE_UPLOAD_ACTION'] = 'ask'
app.config['DROPZONE_UPLOAD_BTN_ID'] = 'submit'
app.config['DROPZONE_DEFAULT_MESSAGE'] = '사진 업로드: 클릭하거나 파일을 드래그해주세요.'
app.config['DROPZONE_MAX_FILE_SIZE'] = 10
##################################################################
dropzone = Dropzone(app)
db = SQLAlchemy(app)
##################################################################
global tempfname
tempfname = ''
##################################################################
#
# """ login information"""
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 465
# SMTP_PASSWORD = ""  # 비밀번호 초기 설정
# pickle.dump(SMTP_PASSWORD, open("pw.pickle", 'wb'))
# SMTP_USER = "findmycatdog@gmail.com"
# SMTP_PASSWORD = pickle.load(open('pw.pickle', 'rb'))
# first = 1

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

""" 사용자 요청 페이지 """
@app.route('/')
@app.route('/find_my_q', methods=['GET', 'POST'])
def ask():
    global tempfname
    # 사용자 요청 발생 시
    if request.method == 'POST':
        if request.files != ImmutableMultiDict([]):
            for key, f in request.files.items():
                if key.startswith('file'):
                    tempfname = f.filename
                    f.save(os.path.join(app.config['UPLOADED_PATH'], tempfname))
            return '', 204
        else:
            # metadata 저장
            ttime = datetime.now()
            time = str(ttime)
            qla = QueryLostAnimals(querydate=time, location=request.form['location'], lostdate=request.form['date'],
                                   animal=request.form['animal'])
            db.session.add(qla)
            db.session.commit()

            # 파일 저장

            fn = (time.replace('-', '.').replace(' ', '_').replace(':', '.') + '_' + str(qla.id) + '.' +
                  tempfname.split('.')[-1])
            print('tempname =' ,tempfname)
            print('fn= ', fn)

            oldname = os.path.join(app.config['UPLOADED_PATH'], tempfname)
            newname = os.path.join(app.config['UPLOADED_PATH'], fn)
            os.rename(oldname, newname)
            qla.filename = fn
            db.session.commit()
            shutil.copy(newname, os.path.join('./static/images/input_image', fn))

            query_path = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list'
            shutil.rmtree(query_path)
            if not os.path.isdir(query_path):
                os.mkdir(query_path)
            shutil.copy(newname, os.path.join(query_path, fn))

            # # email 보내는데 사용할 query image 저장
            # copy_path = 'C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/email'
            # shutil.rmtree(copy_path)
            # if not os.path.isdir(copy_path):
            #     os.mkdir(copy_path)
            # shutil.copy(newname,os.path.join(copy_path, fn))

            # 쿼리 이미지 분류기에 넘기기
            # count = 0
            extract_similar_image_path.main(request.form['location'], request.form['date'], model)

            # filtering된 image re_id 사용할 directory로 copy
            copy_image.main()

            reid_query.main()

            os.remove('./static/images/input_image'+'/'+fn)
            # os.remove('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/query/query_list'+'/'+fn)
            # 작업 소요시간 확인
            print('소요된 시간 :' + str(datetime.now() - ttime))
            # count += 1

            # 결과 페이지로 이동
            return redirect('/find_my_a?id=' + str(qla.id))

    # 기본 화면
    return render_template('index.html')

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
        if len(sims['number'].values) == 0:
            return render_template('find_my_dog_a.html', id=request.args.get('id'), page=page, asked=asked,
                                   found=None, register=register,
                                   pagesize=1)
        else:
            found = dbquery(db='protect_animals_url1', id=tuple(sims['number'].values))
            pagesize = math.ceil(len(found) / ITEMPERPAGE)
            found = sims.merge(pd.DataFrame(pd.DataFrame(found,
                                                         columns=['no', 'number', 'kind', 'color', 'sex', 'neutralization',
                                                                  'age_weight', 'date', 'location', 'characteristic',
                                                                  'deadline', 'center_name', 'center_number',
                                                                  'center_address', 'image', 'url', 'time'])))
            return render_template('find_my_dog_a.html', id=request.args.get('id'), page=page, asked=asked,
                               found=found[found.columns[1:]].values.tolist()[
                                     (page - 1) * ITEMPERPAGE:page * ITEMPERPAGE], register=register, pagesize = pagesize)

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
        # 공고 데이터
        elif db == 'protect_animals_url1':
            cur.execute('SELECT * FROM {} WHERE number in {}'.format(db, id))
        # 사용자 push 요청 목록
        elif db == 'query_need_push':
            cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))

        rows = cur.fetchall()
        conn.close()
        return rows

# send_mail(쿼리 아이디, [공고번호 리스트])
# """ e-mail push """
# def send_mail(queryid, newfoundid):
#     # 사용자 id 및 찾은 공고번호 기반 DB 탐색
#     queryuser = list(dbquery(db='query_need_push', id=queryid)[0])
#     newfound = list(dbquery(db='protect_animals_url1', id=newfoundid))
#
#     # 푸시 알림을 신청한 사용자가 맞을 경우
#     if queryuser[2] == 1:
#         # message header
#         msg = MIMEMultipart()
#         msg["From"] = SMTP_USER
#         msg["To"] = queryuser[1]
#         msg["Subject"] = '찾아줘 CatDog: 유사한 개체가 탐지되었습니다.'
#
#         # message 본문
#         msg.attach(MIMEText(
#             '<html><body><h3>등록하신 개체와 유사한 개체가 탐지되었습니다.</h3><table border="1" bordercolor="#F8F8F8" align = "center"><thead bgcolor="#F4F4F4" align ="center"><tr><th rowspan="2"></th><th>공고번호</th><th>발견일자</th><th>발견장소</th><th colspan="2">공고기간</th><th>보호소</th><th>바로가기</th></tr><tr><th>공고 상 분류</th><th>색상</th><th>성별</th><th>중성화 여부</th><th>추정 연령</th><th>몸무게</th><th>특이사항</th></tr></thead><tbody align ="center">',
#             'html', 'utf-8'))
#         for item in newfound:
#             msg.attach(MIMEText(
#                 '<tr><td rowspan="2"><img src="{}" height=100px></td><td>{}</td><td>{}</td><td>{}</td><td colspan="2">{}</td><td>{}</td><td><a href="{}"><b>공고</b></a></td></tr><tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(item[14], item[1], item[7], item[8], item[10], item[11], item[15], item[2], item[3], item[4], item[5], item[6].split('/')[0][:-1], item[6].split('/')[1][1:], item[9]),
#                 'html', 'utf-8'))
#         msg.attach(MIMEText('</tbody></table></body></html>', 'html', 'utf-8'))
#
#         # 푸시 거부 로직 추가 필요
#
#         # send
#         with smtplib.SMTP_SSL(SMTP_SERVER) as s:
#             s.login(SMTP_USER, SMTP_PASSWORD)
#             s.sendmail(SMTP_USER, msg["To"], msg.as_string())

##################################################################

if __name__ == '__main__':
#    pd.set_option('display.max_rows', None)
#    pd.set_option('display.max_columns', None)
#    pd.set_option('display.width', None)
#    pd.set_option('display.max_colwidth', -1)
    # host = '0.0.0.0', port = 5000
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
