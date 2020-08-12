import os
import time
import socket
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

socket.setdefaulttimeout(30)
driver = webdriver.Chrome('chromedriver.exe')

query = '진돗개'
url = "https://www.google.com/search?q="+query+"&sourceid=chrome&ie=UTF-8&source=lnms&tbm=isch&tbs=itp:photo"
driver.get(url)
driver.maximize_window()

#scroll
last_height = driver.execute_script("return document.body.scrollHeight")
clicked = False

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    new_height = driver.execute_script("return document.body.scrollHeight")
    
    if last_height == new_height:
        if clicked is True:
            break
        else:
            if driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').is_displayed():
                driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
                clicked = True
            elif NoSuchElementException:
                break
    
    last_height = new_height

div = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]')
img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd")

#save
if os.path.isdir(os.path.join('more_data', 'google', query)) is False:
    os.makedirs(os.path.join('more_data', 'google', query))

i = 0
for img in img_list:
    try:   
        img.click()
        driver.implicitly_wait(3)
        src = driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute('src')
        urlretrieve(src, 'more_data/google/{}/{}.jpg'.format(query, i))
        i += 1
    except Exception as e:
        pass

driver.quit()