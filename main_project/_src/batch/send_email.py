import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pymysql
import pickle
import sys
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/dog_image_similarity')
sys.path.append('C:/Users/kdan/BigJob12/main_project/_src/data_analysis/re_id/code')
import extract_similar_image_path, copy_image, predict_dog_data
import reid_query
import pandas as pd
import os

model = predict_dog_data.make_model(256)
sys.path.pop()

""" login information"""
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
#SMTP_PASSWORD = ""  # 비밀번호 초기 설정
#pickle.dump(SMTP_PASSWORD, open("pw.pickle", 'wb'))
SMTP_USER = "findmycatdog@gmail.com"
SMTP_PASSWORD = pickle.load(open('C:/Users/kdan/BigJob12/main_project/_src/web/pw.pickle', 'rb'))
first = 1

# 파이참 실행 시
# query ,email_image 셈플 DB로 넣기
# query가 모델로 들어감
# 맨처음 결과와 다른 이미지(email image) => to_web csv  이랑 지금 파일이랑
# 메일 보내기

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
            if len(id) == 1:
                id += id
                print('SELECT * FROM {} WHERE id= {}'.format(db, id))
                cur.execute('SELECT * FROM {} WHERE number in {}'.format(db, id))
            else:
                print('SELECT * FROM {} WHERE id= {}'.format(db, id))
                cur.execute('SELECT * FROM {} WHERE number in {}'.format(db, id))

        # 사용자 push 요청 목록
        elif db == 'query_need_push':
            cur.execute('SELECT * FROM {} WHERE id= {}'.format(db, id))

        rows = cur.fetchall()
        conn.close()
        return rows
""" e-mail push """
def send_mail(newfoundid, queryid = 365):
    print('-' * 50)
    # 사용자 id 및 찾은 공고번호 기반 DB 탐색
    queryuser = list(dbquery(db='query_need_push', id=queryid)[0])
    if len(newfoundid) == 0: return # 같은게 없을 경우
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
def load_data():
    with open('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/data.pickle', 'rb') as f:
        data = pickle.load(f)
    return data
def compare_list():
    email_list = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_email.csv')['0'].tolist()
    update_list = pd.read_csv('C:/Users/kdan/BigJob12/main_project/_db/data/model_data/working/to_web.csv')['0'].tolist()
    result = []

    for _ in update_list:
        if _ in email_list:
            pass
        else: result.append(_)

    return result

if __name__ == '__main__':
    loc, t = load_data()

    extract_similar_image_path.main(loc, t, model)

    copy_image.main()

    reid_query.main(False)

    compare_lst = compare_list()

    send_mail(tuple(compare_lst))