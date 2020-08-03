import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

# 2008-06-09 이후, 동물보호관리시스템 상 공고 총 702,792건
# https://www.animal.go.kr/front/awtis/public/publicList.do

""" all time """
df1 = pd.read_csv('dogs_breed_list.csv', names=['code', 'name'], dtype={'code': object})
baseurl = "https://www.animal.go.kr/front/awtis/public/publicList.do?searchSDate=2000-08-01&searchEDate=2020-07-31&searchUpKindCd=417000&searchKindCd="
count = []

# 건수
for code in df1.code.values:
    url = baseurl+code
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    a = soup.find_all('ul', {'class':'list'})
    if len(a[0].text[2:-2]) > 8:
        count.append(0)
    else:
        count.append(int(a[0].text[2:-2].replace(',', '')))
df1['count'] = count

# 순위
df1.sort_values('count', ascending=False, inplace = True)
df1['rank'] = range(1, len(df1)+1)

# 비중
ratio1 = []
for i in range(len(df1)):
    if df1['count'].iloc[i] != 0:
        ratio1.append((df1['count'].iloc[i] / sum(df1['count'].values))*100)
    else:
        ratio1.append(0)
df1['ratio'] = ratio1

# 누적 백분위
accum = []
accum.append(0)
for _ in ratio1:
    accum.append(_+accum[-1])
del accum[0]
df1['accum'] = accum

""" Year 2019 """
df3 = pd.read_csv('dogs_breed_list.csv', names=['code', 'name'], dtype={'code': object})
baseurl = "https://www.animal.go.kr/front/awtis/public/publicList.do?searchSDate=2019-01-01&searchEDate=2019-12-31&searchUpKindCd=417000&searchKindCd="
count = []

# 건수
for code in df3.code.values:
    url = baseurl+code
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    a = soup.find_all('ul', {'class':'list'})
    if len(a[0].text[2:-2]) > 8:
        count.append(0)
    else:
        count.append(int(a[0].text[2:-2].replace(',', '')))
df3['count'] = count

# 순위
df3.sort_values('count', ascending=False, inplace = True)
df3['rank'] = range(1, len(df3)+1)

# 비중
ratio1 = []
for i in range(len(df3)):
    if df3['count'].iloc[i] != 0:
        ratio1.append((df3['count'].iloc[i] / sum(df3['count'].values))*100)
    else:
        ratio1.append(0)
df3['ratio'] = ratio1

# 누적 백분위
accum = []
accum.append(0)
for _ in ratio1:
    accum.append(_+accum[-1])
del accum[0]
df3['accum'] = accum

temp = pd.merge(df3, df1, left_on='code', right_on='code').drop('name_y', axis=1)
temp.columns = ['code', 'name', 'count_2019', 'rank_2019', 'ratio_2019', 'accum_2019', 'count_alltime', 'rank_alltime', 'ratio_alltime', 'accum_alltime']
temp = temp[['code', 'name', 'rank_2019', 'rank_alltime', 'ratio_2019', 'ratio_alltime', 'count_2019', 'count_alltime', 'accum_2019', 'accum_alltime']]
temp.to_csv('dogs_public_alltime_and_2019.csv', encoding='cp949')

# 2019년 품종별 비중 (믹스견, 기타 제외)
plt.figure(figsize=(20, 20))
plt.rc('font', family='NanumGothic')
plt.rcParams['font.size'] = 15
plt.pie(df3.drop([4, 40])['count'], labels=df3.drop([4, 40])['name'], startangle=90);
plt.savefig('2019_ratio.png')