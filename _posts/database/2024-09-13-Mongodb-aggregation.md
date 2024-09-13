---
title: MongoDB Aggregation
author: blakewoo
date: 2024-9-13 21:30:00 +0900
categories: [Database]
tags: [Computer science, Database, MongoDB]
render_with_liquid: false
---

# MongoDB aggregation

MongoDB에서 제공하는 가장 강력한 기능이다.
find 기능의 확장판이라고도 볼수 있지만 실상은 원하는 정보를 찾아서 가공하는 집계 함수라고 할 수 있다.

## 장단점
이 기능의 장점과 단점은 아래와 같다.
### 장점
- 단순히 데이터를 찾고 자르는게 그치는 find 함수보다 더 다양한 편집 기능을 지원한다.

### 단점
- 메모리를 많이 요한다. 최소 10MB 만큼의 메모리를 예약하고 시작하며 필요에 따라 기하급수적으로 늘어난다.
- 기본적인 CRUD보다 성능이 좋진 않다.

### 그 외 특징
- 파이프라인 내부 문서의 크기는 최대 64MB까지 가능하다
- 최종 반환 문서의 크기는 BSON 형태로 크기는 16MB 이하여야한다.
- $sort, $group, $bucket, $bucketAuto와 같은 특정 단계는 처리에 디스크 공간을 사용할 수 있으며, 이때 allowDiskUse:true 옵션을 요한다.

## 사용법
시작 자체는 일반적인 CRUD 함수와 다르지 않다.
```javascript
db.cols.aggregation({})
```
해당 aggregation 안에 들어가는 인자가 중요한데 해당 인자 이름을 파이프라인이라고 하며
형태는 배열이다. 각 쿼리를 순차적으로 처리하여 최종 산출물을 output 하는 식이다.

이러한 인자 하나 하나를 스테이지라고 하며 여러 종류를 지원한다.
기본적인 aggregation의 스테이지는 아래와 같다.

```javascript
$match    ==   find(query)
$project  ==   find({},projection) 
$sort     ==   find().sort(order) 
$limit    ==   find().limit(num) 
$skip     ==   find().skip(num) 
$count    ==   find().count()
```

오른쪽에는 해당 스테이지와 동일한 기능을 하는 쿼리를 적어두었다.
find에서 aggregation으로 변경이 가능하기 때문에 이렇게 동일한 쿼리가 존재할 수 있는 것이다.
그래서 review컬렉션을 대상으로 하는 find 쿼리를 aggregation으로 바꾸는 아래와 같은 예시가 가능하다.
```javascript
//find query
db.review.find(
    {"address.country":"Canada"},
    {"host.host_total_listings_count":1,"host.host_name":1}
).sort({"host.host_total_listings_count":-1}).limit(1)

// aggregation
db.review.aggregate([
    {$match:{"address.country":"Canada"}},
    {$sort:{"host.host_total_listings_count":-1 }},
    {$limit:1},
    {$project:{
            "host.host_total_listings_count":1,
            "host.host_name":1}}
])
```

find 쿼리와 aggregation 쿼리를 살펴봤을 때 aggregation의 경우 $가 붙어있음을 알 수 있는데
이렇게 $로 표기하는 뜻이 따로 다 있다.

- 왼쪽에 붙은 $ 하나 : 스테이지 이름을 뜻한다   
  ex) $match
- 콜론(:) 오른쪽에 붙은 $ : 해당 필드의 값이다.   
  ex) {$set: {chicken: "$brand"}} - brand 필드의 값을 chicken 필드에 입력하라
- $multiply는 필드 값 안에 쓰이며 콜론(:) 왼쪽의 필드값을 나타낸다.   
  ex) {$set: {data: {$multiply: [1,4]}}
- $$는 임시 루프를 돌릴때 변수로써 쓰인다.
  ex) $set: {value: {$map: {input: "$value2", as: "i", in: {$multiply :["$$i",4]}}}}}
- $literal은 $ 또는 명시적인 숫자를 포함하는 문자열을 나타낼 수 있다.
  ex) {$set: {dollarPrice: {$literal: "$22.2"}}}

이렇게 $에 대해서 알아보았다.
그런데 단순히 이렇게 find를 대체할 수 있다면 굳이 aggregation을 쓸 필요가 없다고 생각할 수 있다.
당연하지만 이것이 aggregation 기능의 전부가 아니다.
아래를 보자
```
Arithmetic Expression Operators 
Array Expression Operators 
Boolean Expression Operators 
Comparison Expression Operators 
Conditional Expression Operators 
Date Expression Operators
Literal Expression Operator
Object Expression Operators
Set Expression Operators
String Expression Operators
Text Expression Operator 
Trigonometry Expression Operators 
Type Expression Operators 
Accumulators ($group)
```
aggregation에서 지원하는 공식적인 연산자의 카테고리이다.
이렇게나 많지만 우리는 몇가지만 뽑아서 알아보도록 하겠다.

### 세부 기능

#### 1. 산술 연산
- $add : 더하는 stage, 배열 안의 인자들을 더한다
- $subtract : 빼는 stage, 배열 안의 값을 순서대로 뺀다.
- $multiply : 곱하는 stage, 배열 안의 값을 곱한다.
- $divide : 나누는 stage, 배열 안의 값을 나눈다.

ex) 예시들
  ```
  // a 필드값과 b 필드값을 더한다
  { $add : [ "$a", "$b" ] }
  
  // b 필드값에서 a 필드값을 뺀다
  { $subtract : [ "$a", "$b" ] } 
  
  // a 필드값에서 b필드값을 곱한다
  { $multiply : [ "$a", "$b" ] }
  
  // a필드값에서 b필드값을 나눈다.
  { $divide : [ "$a", "$b" ] }
  ```

#### 2. 문자열 연산
- $concat : 배열 안의 인자들을 붙인다.
- $ltrim : 특정 문자를 지정하지 않으면 좌측에 공백을 지운다.
- $indexOfCP : 첫번째 인자에서 두번째 인자 문자가 발견되는 인덱스를 반환함
- $split : 두번째 인자를 기준으로 첫 번째 인자를 배열로 분리함

  ex) 예시들
  ```
  // a필드값과 "_"와 필드값의 문자열을 이은뒤에 반환한다.
  { $concat : [ "$a", "_", "$b" ] }
  
  // a필드값 왼쪽의 공백을 지운다
  { $ltrim : {input: $a} }
  // a필드값 왼쪽에 2가 있으면 지운다
  { $ltrim : { input: $a, chars: "2" } }
  
  // email 필드의 @가 위치한 index를 반환한다.
  { $indexOfCP : [ "$email", "@" ] }
  
  // phone 필드의 값을 "-"를 기준으로 나눈다
  { $split : [ "$phone", "-" ] }
  ```

#### 3. 그룹 연산
사실 aggregation을 쓰는 이유 중 가장 큰 이유라고 봐도 무방하다.

- $group : 들어오는 문서 스트림을 가져와 (SQL에서의 GROUP BY와 가장 유사한) 문서 세트를 결합하여 더 작은 세트로 줄인다.
  해당 GROUP 스테이지에서 지원하는 accumulators가 있다.
  ```
  $addToSet: 각 그룹에 대한 고유한 표현 값의 배열을 추가한다.
  $avg: 숫자 값의 평균을 계산한다.
  $sum: 숫자 값의 합계를 계산한다.
  $first/$last: 각 그룹에 대한 첫 번째 또는 마지막 문서의 값이다.
  $max/$min: 각 그룹에 대한 가장 높은 또는 가장 낮은 표현 값이다.
  $mergeObjects: 각 그룹에 대해 입력 문서로 결합된 문서이다.
  $push: 각 그룹에 대한 표현 값의 배열을 추가한다.
  $stdDevSamp: 입력 값의 표본 표준 편차를 계산한다.
  ```  

  $group을 쓰는 형태는 아래와 같다.
  ```
  {$group: {_id : <expression>,
    field1 : {<$accum>: <expression>},
  …}}
  ```

  ex) 아래의 값이 있다고 가정할때
  ```json
  { "_id": 1, "country": "USA", "city_population": 5000000 },
  { "_id": 2, "country": "Canada", "city_population": 3000000 },
  { "_id": 3, "country": "USA", "city_population": 7000000 },
  { "_id": 4, "country": "Canada", "city_population": 2000000 }
  ``` 
  country 필드 값을 기준으로 city_population를 합산해서 population 필드를 추가하고자하면
  아래의 쿼리를 쓸 수 있다.
  ```javascript
  db.countryInfo.aggregation([
    {$group : {_id: "$country", population :{$sum:"$city_population"}}}
  ])
  ```
  그러면 아래와 같이 결과가 나온다
  ```json
  [
    { "_id": "USA", "population": 12000000 },
    { "_id": "Canada", "population": 5000000 }
  ]
  ```

- $unwind : $group의 반대 작업으로, 어떤 배열 필드에도 적용 가능하며
  하나의 문서를 여러 개로 변환한다.
  아래와 같은 문서가 있다고 하자
  ```
  { a: 1, b: [2,3,4] }
  ```
  b 필드 값에 대해서 분할하고 싶다면 다음과 같은 쿼리를 쓸 수 있다.
  ```javascript
    db.countryInfo.aggregation([
      {$unwind: "$b"}
    ])
  ```
  그러면 아래와 같이 결과가 나온다
  ```json
    {a:1,b:2}, {a:1,b:3}, {a:1,b:4}
  ```

## 몇 가지 사용 예시
위의 사용 설명만 봐서는 아무래도 aggregation에 대해서 이해하기 힘들다. 그래서 사용 예시를 하나 갖고와서
설명하도록 하겠다.
다음 예시는 studio 3t 사이트(https://studio3t.com/ko/knowledge-base/articles/mongodb-aggregation-framework) 에서 제공하는 예시이다.

두가지 컬렉션을 대상으로 하는 예시이며 각 컬렉션의 내용은 아래와 같다.
### UNIVERSITY COLLECTION
```json
{
  country : 'Spain',
  city : 'Salamanca',
  name : 'USAL',
  location : {
    type : 'Point',
    coordinates : [ -5.6722512,17, 40.9607792 ]
  },
  students : [
    { year : 2014, number : 24774 },
    { year : 2015, number : 23166 },
    { year : 2016, number : 21913 },
    { year : 2017, number : 21715 }
  ]
},
{
  country : 'Spain',
  city : 'Salamanca',
  name : 'UPSA',
  location : {
    type : 'Point',
    coordinates : [ -5.6691191,17, 40.9631732 ]
  },
  students : [
    { year : 2014, number : 4788 },
    { year : 2015, number : 4821 },
    { year : 2016, number : 6550 },
    { year : 2017, number : 6125 }
  ]
}
```

### COURSE COLLECTION
```JSON
{
  university : 'USAL',
  name : 'Computer Science',
  level : 'Excellent'
},
{
  university : 'USAL',
  name : 'Electronics',
  level : 'Intermediate'
},
{
  university : 'USAL',
  name : 'Communication',
  level : 'Excellent'
}
```

### 예시 1
group 스테이지를 이용하여 개수, 합계, 평균 또는 최대값 찾기와 같이 필요한 모든 집계 또는 요약 쿼리를 수행할 수 있다.
가령 해당 university name에 대해 document 수를 알고 싶을때는 아래와 같은 쿼리를 쓸수 있다.
```javascript
db.universities.aggregate([
  { $group : { _id : '$name', totaldocs : { $sum : 1 } } }
]).pretty()
```

name 필드값의 갯수 합계를 구하되 totaldocs 필드에 해당 갯수를 나타내고 싶을 때 사용하는 쿼리다.

결과
```json
{ "_id" : "UPSA", "totaldocs" : 1 }
{ "_id" : "USAL", "totaldocs" : 1 }
```



### 예시 2
대학이름이 USAL인 대학에 대해서 student단위로 document를 분할 하고 싶다면 아래와 같이 쿼리를 쓸 수 있다.
```javascript
db.universities.aggregate([
  { $match : { name : 'USAL' } },
  { $unwind : '$students' }
]).pretty()
```

이름이 USAL인 document를 분할하는데 students 기준으로 document를 분할한다.

결과
```json
{
	"_id" : ObjectId("5b7d9d9efbc9884f689cdba9"),
	"country" : "Spain",
	"city" : "Salamanca",
	"name" : "USAL",
	"location" : {
		"type" : "Point",
		"coordinates" : [
			-5.6722512,
			17,
			40.9607792
		]
	},
	"students" : {
		"year" : 2014,
		"number" : 24774
	}
}
{
	"_id" : ObjectId("5b7d9d9efbc9884f689cdba9"),
	"country" : "Spain",
	"city" : "Salamanca",
	"name" : "USAL",
	"location" : {
		"type" : "Point",
		"coordinates" : [
			-5.6722512,
			17,
			40.9607792
		]
	},
	"students" : {
		"year" : 2015,
		"number" : 23166
	}
}
{
	"_id" : ObjectId("5b7d9d9efbc9884f689cdba9"),
	"country" : "Spain",
	"city" : "Salamanca",
	"name" : "USAL",
	"location" : {
		"type" : "Point",
		"coordinates" : [
			-5.6722512,
			17,
			40.9607792
		]
	},
	"students" : {
		"year" : 2016,
		"number" : 21913
	}
}
{
	"_id" : ObjectId("5b7d9d9efbc9884f689cdba9"),
	"country" : "Spain",
	"city" : "Salamanca",
	"name" : "USAL",
	"location" : {
		"type" : "Point",
		"coordinates" : [
			-5.6722512,
			17,
			40.9607792
		]
	},
	"students" : {
		"year" : 2017,
		"number" : 21715
	}
}
```

## 개인적인 TIP

1. MongoDB에 Aggregation을 써보기 위한 가장 좋은 툴은 COMPASS이다.
- MongoDB에서 공식적으로 제공하는 것으로 각 스테이지 별 예시 표기를 지원한다.
- 실제 MongoDB 공식 교육에서도 해당 툴을 사용한다.

2. Aggregation을 쓸때는 $match 스테이지로 범위를 줄이고 시작하라.
- match, project 나 project, match나 결과는 같지만 대상의 크기가 달라져서 성능이 달라진다.




# 참고자료
- [MongoDB 공식 문서](https://www.mongodb.com/ko-kr/docs/manual/core/document/)

