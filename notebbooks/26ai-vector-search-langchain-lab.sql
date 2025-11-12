-- SQL Worksheet로 DB에 접속하여 다음을 실행합니다.

-- 기생성된 테이블이 있는 경우 삭제합니다.
drop table DOCUMENTS_COSINE;

-- OracleVS로 생성된 테이블을 확인합니다.
desc DOCUMENTS_COSINE;

-- vector_store.add_documents() 결과로 추가된 문서 청크를 확인합니다.
SELECT id, text, to_char(METADATA), EMBEDDING FROM DOCUMENTS_COSINE;

SELECT to_char(METADATA) FROM DOCUMENTS_COSINE;

SELECT JSON_SERIALIZE(METADATA PRETTY) AS pretty_json
FROM DOCUMENTS_COSINE
WHERE ROWNUM<=3;

-- response = chain.invoke(question) 코드 실행으로 인해 DB에서 실행한 SQL 쿼리를 확인하는 질의입니다.
SELECT sql_id, parsing_schema_name, sql_text
FROM v$sql
WHERE parsing_schema_name = 'VECTOR' and module like '%python'
ORDER BY last_active_time DESC;

-- 실행한 SQL 쿼리에 대한 실행 계획을 확인합니다.
EXPLAIN PLAN FOR
    SELECT id, text, metadata, vector_distance(embedding, :embedding, COSINE) as distance
      FROM DOCUMENTS_COSINE
  ORDER BY distance
     FETCH APPROX FIRST 4 ROWS ONLY;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY(format => 'ALL'));

-- 생성된 HNSW Index를 확인합니다.
SELECT * FROM ALL_INDEXES WHERE table_name=UPPER('DOCUMENTS_COSINE');

-- 실행한 SQL 쿼리에 대한 실행 계획이 달라졌는지 확인합니다.
EXPLAIN PLAN FOR
    SELECT id, text, metadata, vector_distance(embedding, :embedding, COSINE) as distance
      FROM DOCUMENTS_COSINE
  ORDER BY distance
     FETCH APPROX FIRST 4 ROWS ONLY;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY(format => 'ALL'));

-- vector_store.similarity_search(question, 3, filter=filter_dict) 같이 필터링을 사용했을 때 실행되는 SQL을 확인합니다.
SELECT sql_id, parsing_schema_name, sql_text
FROM v$sql
WHERE parsing_schema_name = 'VECTOR' and module like '%python' and sql_text like '%DOCUMENTS_COSINE%'
ORDER BY last_active_time DESC;

-- JSON 타입인 metadata에 대해서 추가적인 인덱스 설정이 필요합니다.
CREATE SEARCH INDEX metadata_json_search_idx
ON DOCUMENTS_COSINE (metadata)
FOR JSON;

-- 인덱스 생성후 실행계획을 확인해 보면 새로운 JSON 인덱스를 사용하는 것을 알 수 있습니다.
EXPLAIN PLAN FOR
  SELECT id, text, metadata, vector_distance(embedding, :embedding, COSINE) as distance
    FROM "DOCUMENTS_COSINE"
   WHERE JSON_EXISTS(metadata, '$.source?(@ == $val)' PASSING :value0 AS "val")
ORDER BY distance
   FETCH APPROX FIRST 3 ROWS ONLY;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY(format => 'ALL'));