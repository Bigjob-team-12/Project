import os
import time
import socket
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

<<<<<<< HEAD
<<<<<<< HEAD
socket.setdefaulttimeout(30) # 장기 미응답 처리
driver = webdriver.Chrome('chromedriver.exe')

# selenium - google image search
query = '진돗개' # 검색어
=======
socket.setdefaulttimeout(30)
driver = webdriver.Chrome('chromedriver.exe')

query = '진돗개'
>>>>>>> 6f1b3ba... dataset collection
=======
socket.setdefaulttimeout(30) # 장기 미응답 처리
driver = webdriver.Chrome('chromedriver.exe')

# selenium - google image search
query = '진돗개' # 검색어
>>>>>>> 1e41458... comment
url = "https://www.google.com/search?q="+query+"&sourceid=chrome&ie=UTF-8&source=lnms&tbm=isch&tbs=itp:photo"
driver.get(url)
driver.maximize_window()

<<<<<<< HEAD
<<<<<<< HEAD
# scroll down
last_height = driver.execute_script("return document.body.scrollHeight")
clicked = False
=======
#scroll
last_height = driver.execute_script("return document.body.scrollHeight")
clicked = False

>>>>>>> 6f1b3ba... dataset collection
=======
# scroll down
last_height = driver.execute_script("return document.body.scrollHeight")
clicked = False
>>>>>>> 1e41458... comment
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    if last_height == new_height:
        if clicked is True:
            break
        else:
<<<<<<< HEAD
<<<<<<< HEAD
            if driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').is_displayed(): # 결과 더보기 버튼 클릭
=======
            if driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').is_displayed():
>>>>>>> 6f1b3ba... dataset collection
=======
            if driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').is_displayed(): # 결과 더보기 버튼 클릭
>>>>>>> 1e41458... comment
                driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
                clicked = True
            elif NoSuchElementException:
                break
    
    last_height = new_height

<<<<<<< HEAD
<<<<<<< HEAD
# 개별 이미지 탐색
div = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]') # 전체 이미지 영역
img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd") # 개별 이미지 영역

# 저장 폴더 생성
if os.path.isdir(os.path.join('more_data', 'google', query)) is False:
    os.makedirs(os.path.join('more_data', 'google', query))

# 각 이미지 클릭 후 원본 이미지 url에서 이미지 download
=======
div = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]')
img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd")
=======
# 개별 이미지 탐색
div = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]') # 전체 이미지 영역
img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd") # 개별 이미지 영역
>>>>>>> 1e41458... comment

# 저장 폴더 생성
if os.path.isdir(os.path.join('more_data', 'google', query)) is False:
    os.makedirs(os.path.join('more_data', 'google', query))

<<<<<<< HEAD
>>>>>>> 6f1b3ba... dataset collection
=======
# 각 이미지 클릭 후 원본 이미지 url에서 이미지 download
>>>>>>> 1e41458... comment
i = 0
for img in img_list:
    try:   
        img.click()
        driver.implicitly_wait(3)
<<<<<<< HEAD
<<<<<<< HEAD
        src = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute('src') # 원본 url
        urlretrieve(src, 'more_data/google/{}/{}.jpg'.format(query, i)) #저장
=======
        src = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')
        urlretrieve(src, 'more_data/google/{}/{}.jpg'.format(query, i))
>>>>>>> 6f1b3ba... dataset collection
=======
        src = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute('src') # 원본 url
        urlretrieve(src, 'more_data/google/{}/{}.jpg'.format(query, i)) #저장
>>>>>>> 1e41458... comment
        i += 1
    except Exception as e:
        pass

driver.quit()