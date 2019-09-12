# NLP_similarity

- 텍스트가 얼마나 유사한지를 표현하는 방식
- 다른 구조의 문장이지만, 의미가 비슷하면 두 문장의 유사도가 높다
- 데이터를 정량화 하는 것이 중요

> 데이터 정량화하는 방법
-	1. 단순히 텍스트를 벡터화한 후 벡터화된 각 문장 간의 유사도를 측정 
-	2. 같은 단어의 개수를 사용해서 유사도를 판단하는 방법
-	3. 형태소로 나누어 형태소를 비교하는 방법
-	4. 자소 단위로 나누어 단어를 비교하는 방법 등

데이터 : kaggle : https://www.kaggle.com/c/quora-question-pairs/data

다운로드
~~~
kaggle competitions download -c quora-question-pairs
~~~
  - 만약, 403 error가 나면 (https://www.kaggle.com/c/quora-question-pairs/rules) 여기서 accept를 해주면 됨. 
  
#### 데이터 설명
질문과 답변을 할 수 있는 사이트 --> 중복된 질문들을 잘 찾기만 한다면 이미 잘 작성된 답변들을 사용자들이 참고하게 할 수 있음

## 데이터 분석과 전처리
- 그 데이터 분석을 통해 ***데이터 특징 파악***
- 특징을 파악 후, 데이터 전처리 작업 진행

#### 데이터 zip 파일 풀기
~~~
import zipfile

DATA_IN_PATH = './data_in/'

file_list = ['train.csv.zip', 'test.csv.zip', 'sample_submission.csv.zip']

for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()
~~~

> 데이터 설명
> train : 학습 데이터
> test : 평가 데이터
> sample_submission : 모델을 통해 평가 데이터에 대한 예측 겨로가를 캐글 홈페이지에 제출할 때 양식을 맞추기 위해 보여주는 예시 파일

