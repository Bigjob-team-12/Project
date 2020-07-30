from bs4 import BeautifulSoup
import pymysql
import requests
import re
from datetime import datetime

# DB connect
conn = pymysql.connect(host='localhost', user='root', password='1234',
                   db='project', charset='utf8')

def download(url, params={}, method='GET', headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}):
    '''
    url crawling
    :param url: 요청받을 url
    :param params: key
    :param headers:
    :param method:
    :param limit:
    :return: response
    '''
    resp = requests.request(method, url,
                            params=params if method == 'GET' else {},
                            data=params if method == 'POST' else {},
                            headers=headers)

    return resp

def protect_animals_url1_to_urllist(url, page_range):
    '''
    동물보호관리시스템 유기동물 공고에서 page range만큼 url crawling
    :param url: Crawling할 첫 page
    :param page_range: 페이지 범위
    :return: Scraping 해야 할 url list
    '''
    print('-' * 10, ' protect_url1 Crawling start  ', '-' * 10)
    urllist = []
    for page in page_range:
        resp = download(url=url, params={'page': page}, method='GET')
        [urllist.append(url.replace('list', 'view') + '?' + _) for _ in
         re.findall(r'na_open_window\(\'win\', \'abandonment_view_api.php\?([^\"]+)', resp.text)]

    print('-' * 10, ' protect_url1 Crawling end  ', '-' * 10)
    return urllist
def protect_animals_url1_check_data_existence(data):
    '''
    해당 data 유무 check
    :param data:
    :return: 데이터 유무 : 1 or 0
    '''
    curs = conn.cursor()
    # 해당 데이터 존재 유무 확인
    sql = 'SELECT count(*) from protect_animals_url1 where number like "' + data['공고번호'] + '"'

    # insert data
    curs.execute(sql)

    rows = curs.fetchall()
    return rows[0][0]
def protect_animals_url1_scraping(urllist):
    '''
    image & 필요한 정보 scraping
    :param urllist: Scraping 할 urllist
    :return: dict 형태의 data
    '''
    print('-' * 10, ' protect_url1 Scraping start  ', '-' * 10)

    result = []

    for url in urllist:
        data = dict()
        resp = download(url=url, method='GET')
        dom = BeautifulSoup(resp.text, 'lxml')
        tmp = dom.select('table.viewTable th, table.viewTable td')
        tmp = tmp[1:21] + tmp[-15:3]

        for i in range(0, len(tmp), 2):
            data[tmp[i].text.strip()] = tmp[i + 1].text.strip()
        data['나이'] = data['나이/체중'][:4]
        data['체중'] = data['나이/체중'][11:]
        del data['나이/체중']

        # DB에 data 유무 확인
        if protect_animals_url1_check_data_existence(data):
            break

        tmp = dom.select('strong')
        data['보호센터이름'] = tmp[0].text
        data['보호센터전화번호'] = tmp[1].text
        data['보호장소'] = tmp[2].text

        data['image'] = dom.select_one('img')['src']

        data['url'] = url
        data['time'] = datetime.now()

        result.append(data)

    print('-' * 10, ' protect_url1 Scraping end  ', '-' * 10)
    return result
def protect_animals_url1_to_DB(data):
    '''
    protect_url1_result to DB
    :param data:
    :return:
    '''
    print('-' * 10, ' protect_url1 to_database start  ', '-' * 10)
    # DB connect
    curs = conn.cursor()

    sql = '''INSERT INTO protect_animals_url1(number, kind, color, sex, neutralization, date, location, characteristic, deadline, age, weight, center_name, center_number, center_address, image, url, time)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''

    # insert data
    [curs.execute(sql, (tuple(_.values()))) for _ in data]
    conn.commit()
    print('-' * 10, ' protect_url1 to_database end  ', '-' * 10)
def protect_animals_url1(page):
    '''
    동물보호관리시스템 유기동물 공고의 데이터 수집
    :param page: 원하는 페이지 수
    :return:
    '''
    # 동물보호관리시스템 유기동물 공고
    protect_url1 = 'http://www.zooseyo.or.kr/Yu_abandon/abandonment_list_api.php'
    page_range = range(page)

    # url crawling
    protect_url1_list = protect_animals_url1_to_urllist(protect_url1, page_range)
    #     print([print(_) for _ in protect_url1_list])

    # urllist scraping
    protect_animals_url1_result = protect_animals_url1_scraping(protect_url1_list)


    [print(_) for _ in protect_animals_url1_result]
    print('protect_animals_url1에서 가지고 온 data 갯수 : ', len(protect_animals_url1_result))

    # protect_url1_result to DB
    protect_animals_url1_to_DB(protect_animals_url1_result)

def protect_animals_url2_find_no(dom):
    '''
    동물보호관리시스템 유기동물 공고에 가장 최근에 올라온 게시글의 'no' 찾는 함수
    :param dom: 
    :return: 최근 게시글 번호
    '''
    # find link
    urllist = dom.select('a')
    # find 최근 no
    for _ in urllist:
        if re.findall(r'no=([0-9]+)', _['href']):
            no = int(re.findall(r'no=([0-9]+)', _['href'])[0])
            break
            
    return no
def protect_animals_url2_scraping(no, n):
    '''
    최근 게시물부터 n개 scraping
    :param no: 최근 게시물 number
    :param n: 수집할 데이터 수
    :return: dict 형태의 data
    '''
    print('-' * 10, ' protect_url2 Scraping start  ', '-' * 10)
    # no부터 n번 돌면서 각 정보 Scraping
    result = []
    url = 'http://www.zooseyo.or.kr/Yu_board/petcare_view.html'

    print(no)
    for num in range(no, no - n, -1):
        resp = download(url=url, params={'no': num}, method='GET')
        dom = BeautifulSoup(resp.text, 'html.parser')

        data = dict()
        # 찾은 동물은 pass
        if not re.findall(r'pet_find_phoneblind_img', resp.text):
            # text = dom.select('td:nth-of-type(1) td')#[63].select('td')[1]#.text.split('\n')
            # text = dom.select('td:nth-of-type(1) td td')

            text = dom.find('td', {'bgcolor' : '#FFFFFF'}) # .select_one('table > tr > td:nth-of-type(2)')

            data['no'] = num
            data['name'] = text.select_one('table > tr > td:nth-of-type(2)').text
            data['date'] = text.select_one('table > tr > td:nth-of-type(4)').text
            data['phone_num'] = text.select_one('table > tr:nth-of-type(3) tr > td:nth-of-type(2)').text
            data['sex'] = text.select_one('table > tr:nth-of-type(3) tr > td:nth-of-type(4)').text
            data['address'] = text.select_one('table > tr:nth-of-type(5) tr > td:nth-of-type(2)').text
            data['title'] = text.select_one('table > tr:nth-of-type(7) tr > td:nth-of-type(2)').text
            data['text'] = text.select_one('table > tr:nth-of-type(9) p').text

            # 글제목 / 내용 / 이미지
            print(text.select_one('table > tr:nth-of-type(11) img'))
            print(len(text.select('table > tr:nth-of-type(11) img')))
            print(data)


            break

            data['image'] = str(['http://www.zooseyo.or.kr' + _ for _ in
                             set(re.findall(r'\/pet_care\/photo\/[0-9_]+.jpe?g', resp.text))])

            text = dom.select('p')

            data['text'] = text[0].text.strip()

            # print(dom.select('p > br')[1].text.strip(), dom.select('p > br')[0].text.strip())
            result.append(data)

    print('-' * 10, ' protect_url2 Scraping end  ', '-' * 10)
    return result
def protect_animals_url2_to_DB(data):
    '''
    protect_url2_result to DB
    :param data:
    :return:
    '''
    print('-' * 10, ' protect_url2 to_database start  ', '-' * 10)
    # DB connect
    conn = pymysql.connect(host='localhost', user='root', password='1234',
                           db='project', charset='utf8')
    curs = conn.cursor()

    sql = '''INSERT INTO protect_animals_url2(number, name, date, phone_num, sex, address, image, characteristic)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'''

    # insert data
    [curs.execute(sql, (tuple(_.values()))) for _ in data]
    conn.commit()
    print('-' * 10, ' protect_url2 to_database end  ', '-' * 10)
def protect_animals_url2(n):
    '''
    유기견보호센터 유기동물 보호중의 데이터 수집
    :param n: 수집할 데이터 수
    :return:
    '''
    print('-' * 10, ' protect_animals_url2 Crawling start ', '-' * 10)
    # 유기견보호센터 유기동물 보호중
    protect_url2 = 'http://www.zooseyo.or.kr/Yu_board/petcare.html'

    resp = download(url=protect_url2,params={'page': 1}, method='GET')
    dom = BeautifulSoup(resp.text, 'html.parser')

    # find 최근 게시글 no
    no = protect_animals_url2_find_no(dom)

    # 최근 게시글부터 n개 Scraping
    protect_animals_url2_result = protect_animals_url2_scraping(no, n)

    print('-' * 10, ' protect_animals_url2 Crawling end ', '-' * 10)
    # to DB
    [print(_) for _ in protect_animals_url2_result]
    print('protect_animals_url2에서 가지고 온 data 갯수 : ', len(protect_animals_url2_result))

    # protect_animals_url2_to_DB(protect_animals_url2_result)

def missing_animals_url3_find_no(dom):
    '''
    동물보호관리시스템 실종동물 공고에 가장 최근에 올라온 게시글의 'no' 찾는 함수
    :param dom:
    :return: 최근 게시글 번호
    '''
    # find 최근 no
    text = dom.find('table', {'background': '../images/board/main-search-img-frame-01.gif'})
    # print(re.findall(r'petfind_view_skin_1.html\?no=[0-9]+', str(text)))

    no = int(re.findall(r'petfind_view_skin_[12].html\?no=([0-9]+)', str(text))[0])
    return no
def missing_animals_url3_scraping(no, n):
    '''
    최근 게시물부터 n개 scraping
    :param no: 최근 게시물 number
    :param n: 수집할 데이터 수
    :return: dict 형태의 data
    '''
    print('-' * 10, ' missing_url3 Scraping start  ', '-' * 10)
    # no부터 n번 돌면서 각 정보 Scraping
    result = []
    url = 'http://www.zooseyo.or.kr/Yu_board/petfind_view_skin_1.html'


    for num in range(no, no - n, -1):
        resp = download(url=url, params={'no': num}, method='GET')
        dom = BeautifulSoup(resp.text, 'html.parser')

        data = dict()
        # 찾은 동물은 pass
        if not re.findall(r'감사합니다\. 찾았어요\!', resp.text):
            text = dom.select('b')

            data['no'] = num
            data['phone_num'] = '-'.join(re.findall(r'[0-9]{3,4}',text[0].text))
            data['address'] = text[1].text
            data['date'] = text[2].text
            data['sex'] = text[5].text.strip()
            data['image'] = str(['http://www.zooseyo.or.kr' + _ for _ in
                             set(re.findall(r'\/pet_care\/photo\/[0-9_]+.jpe?g', resp.text))])
            data['text'] = text[6].text.strip().replace('\r\n',' ')



            result.append(data)

    print('-' * 10, ' missing_url3 Scraping end  ', '-' * 10)
    return result
def missing_animals_url3_to_DB(data):
    '''
    missing_animals_url3 to DB
    :param data:
    :return:
    '''
    print('-' * 10, ' missing_animals_url3 to_database start  ', '-' * 10)
    # DB connect
    conn = pymysql.connect(host='localhost', user='root', password='1234',
                           db='project', charset='utf8')
    curs = conn.cursor()

    sql = '''INSERT INTO missing_animals_url3(number, phone_num, address, date, sex, image, characteristic)
            VALUES (%s,%s,%s,%s,%s,%s,%s)'''

    # insert data
    [curs.execute(sql, (tuple(_.values()))) for _ in data]
    conn.commit()
    print('-' * 10, ' missing_animals_url3 to_database end  ', '-' * 10)
def missing_animals_url3(n):
    '''
    유기견보호센터 실종동물 데이터 수집
    :param n: 수집할 데이터 수
    :return:
    '''
    print('-' * 10, ' missing_animals_url3 Crawling start ', '-' * 10)
    # 유기견보호센터 실종동물 찾는 url
    protect_url2 = 'http://www.zooseyo.or.kr/Yu_board/petfind.html'

    resp = download(url=protect_url2,params={'page': 1}, method='GET')
    dom = BeautifulSoup(resp.text, 'html.parser')

    # find 최근 게시글 no
    no = missing_animals_url3_find_no(dom)

    # 최근 게시글부터 n개 Scraping
    missing_animals_url3_result = missing_animals_url3_scraping(no, n)


    # protect_url2_result = missing_animals_url3_scraping(no, n)
    #
    print('-' * 10, ' missing_animals_url3 Crawling end ', '-' * 10)
    # to DB
    [print(_) for _ in missing_animals_url3_result]
    print('missing_animals_url3에서 가지고 온 data 갯수 : ', len(missing_animals_url3_result))

    missing_animals_url3_to_DB(missing_animals_url3_result)

if __name__ == '__main__':
    # '동물보호관리시스템 유기동물 공고' url의 데이터 수집
    # protect_animals_url1(page = 10)

    # # '유기견보호센터 유기동물 보호중' url의 데이터 수집
    protect_animals_url2(30)

    # # '유기견보호센터 실종동물' url의 데이터 수집
    # missing_animals_url3(30)