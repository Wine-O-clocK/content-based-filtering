# -*- coding: utf-8 -*-
"""wine_cf_algo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dn2BftUXQ5jX3W2nnaU0iZ0NbV5VveVL
"""


import pandas as pd
import numpy as np
import numpy as np

pd.options.display.max_columns=100
pd.options.display.max_rows=1003

#데이터터 읽어오기
#wines=pd.read_csv('wine_all_data.csv')
wine=pd.read_csv('wine_white.csv')

#wine.shape
wine

"""와인 데이터 기본 컬럼명 </br>
[이미지, 이름, 영문이름, 종류, 가격, 당도, 바디, 품종, 아로마1, 아로마2, 아로마3]
</br>
추가 데이터 </br>
[접근성, 이벤트, 홈술] 
"""

#유효하게 사용되어질 컬럼들을 더해서 만든 'feature'

#와인이름 빈공간 없이 만들기
wine['이름'] = wine['이름'].str.replace(' ','')

#int -> str 바꾸기 
wine['가격'] = wine['가격'].astype(str)
wine['당도'] = wine['당도'].astype(str)
wine['바디'] = wine['바디'].astype(str)

#키워드 특징 모두 문자열로 합치기
wine['키워드'] = wine['종류'] + ' ' + wine['가격'] + '만원대 ' + wine['당도'] + '당도 ' + wine['바디'] + '바디 ' + wine['품종'] + ' ' + wine['아로마1'] + ' ' + wine['아로마2'] + ' ' + wine['아로마3']

#컬럼간 더하기 연산을 할 때 값중에 한 개라도 nan 값이 있으면 더해진 값은 무조건 nan으로 반환되는 걸 방지하기 위함
#wine['feature'] = wine['특징'].str.replace('0',' ')
wine['키워드'].head()

"""문제점</br>
- 당도, 가격, 바디 가 숫자형이라 태그로 추천할 때 제대로 먹히지 않음
- NaN 이 있는 경우 모두 합쳣을때 NaN이 됨

TfidfVectorizer 를 import
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# min_df를 1로 설정해줌으로써 한번이라도 노출이 된 정보도 다 고려함
# ngram_range : n_gram 범위 지정 연속으로 나오는 단어들의 순서도 고려함
tf = TfidfVectorizer(min_df=1,ngram_range=(1,5))
tfidf_matrix = tf.fit_transform(wine['키워드'])
tfidf_matrix

#유사도 살펴보기
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_sim

#와인이름을 index로 한 Series 생성
wine_index = pd.Series(data=wine.index, index=wine['이름'])
wine_index.head()

#함수제작
def get_recommed_wine(name):
  names = []
  #선택한 와인의 이름으로부터 해당되는 인덱스를 가져옴
  #선택한 와인을 가지고 연산이 가능해짐
  idx = wine_index[name]
  
  #모든 와인에 대해서 해당 와인과의 유사도를 구함
  sim_scores = list(enumerate(cosine_sim[idx]))

  #유사도에 따라 와인 정렬
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  #가장 유사한 10개의 와인 가져옴
  sim_scores = sim_scores[1:11]

  #가장 유사한 10개의 영화의 인덱스 가져옴
  wine_indices = [i[0] for i in sim_scores]

  #가장 유사한 10개의 와인의 이름 리턴
  return wine_index[wine_indices]

get_recommed_wine('낙낙화이트블랜드') #여기에 자신의 먹어본/저장한 와인을 넣으면 그 와인과 비슷한 와인을 추천해줍니다!

#무엇이 비슷한지 한 번 확인해보기
wine.loc[[207, 1290, 2201, 2224, 1800, 318, 320, 372, 2359, 2329], ['이름', '키워드']]
