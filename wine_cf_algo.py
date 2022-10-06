import pandas as pd
import numpy as np
import numpy as np

pd.options.display.max_columns=100
pd.options.display.max_rows=1003

#데이터터 읽어오기
#wines=pd.read_csv('wine_all_data.csv')
wine_rose=pd.read_csv('wine_rose.csv')

wine_rose.shape
wine_rose

#유효하게 사용되어질 컬럼들을 더해서 만든 'feature'

#와인이름 빈공간 없이 만들기
wine_rose['1'] = wine_rose['1'].str.replace(' ','')

# 컬럼간 더하기 연산을 할 때 값중에 한 개라도 nan 값이 있으면 더해진 값은 무조건 nan으로 반환되는 걸 방지하기 위함
# 3(종류) 4(가격대) 5(당도) 6(바디) 7(아로마1) 8(아로마2) 9(아로마3)
wine_rose['feature'] = wine_rose['3'] + ' ' + wine_rose['7'] + ' ' + wine_rose['8']
wine_rose['feature'] = wine_rose['feature'].str.replace('0',' ')
wine_rose['feature'].head()


# min_df를 1로 설정해줌으로써 한번이라도 노출이 된 정보도 다 고려함
# ngram_range : n_gram 범위 지정 연속으로 나오는 단어들의 순서도 고려함
tf=TfidfVectorizer(min_df=1,ngram_range=(1,5))
tfidf_matrix=tf.fit_transform(book['feature'])
tfidf_matrix
