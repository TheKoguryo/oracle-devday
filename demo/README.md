# Install Streamlit

1. steamlit을 설치합니다.

    ```
    pip install streamlit
    streamlit hello
    ```

2. OS 방화벽을 엽니다.

    ```
    sudo firewall-cmd --permanent --add-port=8501/tcp
    sudo firewall-cmd --reload
    ```

3. 서브넷의 Security List를 엽니다.

    - Source CIDR: 0.0.0.0/0
    - Protocol: TCP
    - Destination Port Range: 8501
    - Description: for Streamlit

# App 설정

1. app.env.template 파일을 복사하여, app.env 파일을 생성합니다.
2. app.env 내 필요한 값을 입력합니다.
3. config.yaml 에서 로그인시 사용할 유저정보 또는 패스워드만 수정합니다.
4. 필요한 라이브러리 설치

    ```
    pip install -r requirements.txt
    ```

5. 앱 실행 

    ```
    streamlit run app-streamlit.py
    ```

6. 로그인 - kildong / xxx

# 테스트 문서

[SPRi 소프트웨어정책연구소](https://spri.kr) 사이트에서 테스트 문서를 다운로드 받을 수 있습니다.
- [AI 브리프] 2023년 12월호: https://spri.kr/posts/view/23669?code=AI-Brief&s_year=&data_page=3
- [AI 브리프] [AI Brief 스페셜] AI 데이터센터 동향과 시사점: https://spri.kr/posts/view/23896?code=AI-Brief&s_year=&data_page=1
- [산업 연간보고서] 2023년 소프트웨어산업 연간보고서: https://spri.kr/posts/view/23733?code=annual_reports&s_year=&data_page=1

# 참고 SQL

```sql
-- 생성된 테이블 확인
DESC ORAVS_DOCUMENTS;

-- 추가된 데이터 확인
SELECT ID, TEXT, json(json_serialize(metadata)), EMBEDDING FROM ORAVS_DOCUMENTS WHERE ROWNUM<=5;

-- 호출된 쿼리 확인
SELECT sql_id, parsing_schema_name, sql_text
FROM v$sql
WHERE parsing_schema_name = 'VECTOR' and module like '%python%' and sql_text like '%ORAVS_DOCUMENTS%'
ORDER BY last_active_time DESC;

-- 인덱스 생성
CREATE SEARCH INDEX metadata_json_search_idx
ON ORAVS_DOCUMENTS (metadata)
FOR JSON;

-- 플랜 확인
EXPLAIN PLAN FOR
  SELECT id, text, metadata, vector_distance(embedding, :embedding, COSINE) as distance
    FROM "ORAVS_DOCUMENTS"
   WHERE JSON_EXISTS(metadata, '$.source?(@ == $val)' PASSING :value0 AS "val")
ORDER BY distance
   FETCH APPROX FIRST 3 ROWS ONLY;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY(format => 'ALL'));
```