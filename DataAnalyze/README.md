# Data Tool 사용법
## 목차
1. [Flight 데이터 수집 과정](#🛬Flight-데이터-수집-과정)
2. [Flighet 데이터 Generator 사용 방법](#)
---
## 🛬Flight 데이터 수집 과정
### 💺출도착 데이터
[Download page](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr)
- 수집경로 : Bureau of Transportation Statistics Dataset
- 년도와 월을 선택가능 (2023년도 8월 기준 2023년도 5월까지 데이터 다운로드 가능)
- 사용하는 정보
    | 정보 | 설명 | Lookup Table |
    | --- | --- | --- |
    | FlightDate | Flight Date (yyyymmdd) |  |
    | Marketing_Airline_Network | Unique Marketing Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years. | [Link](https://drive.google.com/file/d/1rAW1EWRamB0zbKbvbS714mbpcm1FgH31/view?usp=drive_link) |
    | Operating_Airline | Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years. | [Link](https://drive.google.com/file/d/1rAW1EWRamB0zbKbvbS714mbpcm1FgH31/view?usp=drive_link) |
    | Tail_Number | Tail Number |  |
    | Origin | Origin Airport | [Link](https://drive.google.com/file/d/1FR1t-Wx_-DLfZuFBW2l4I1veB9lcQd4d/view?usp=drive_link) |
    | Dest | Destination Airport | [Link](https://drive.google.com/file/d/1FR1t-Wx_-DLfZuFBW2l4I1veB9lcQd4d/view?usp=drive_link) |
    | CRSDepTime | CRS Departure Time (local time: hhmm) |  |
    | CRSArrTime | CRS Arrival Time (local time: hhmm) |  |
    | CRSElapsedTime | CRS Elapsed Time of Flight, in Minutes |  |
    | Distance | Distance between airports (miles) |  |

### ✈ Tailnum에 따른 Aircraft Model 데이터
- 수집경로 : SFO(SanFrancisco International Airport)
    [Aircraft Tail Numbers and Models at SFO | DataSF | City and County of San Francisco](https://data.sfgov.org/w/u7dr-xm3v/ikek-yizv?cur=Vj-QDZbm7Lu&from=root)
- 수집 이유 : BTS 데이터 셋에는 Tailnum만 존재하지 Aircraft Model에 대한 정보는 존재하지 않음. 하지만 crew pairing의 경우 하나의 pairing이 유사한 Aircraft Model를 기반으로 형성되므로 Aircraft 모델 정보가 필요함.

---
## 🛬FlightDataGenerator 사용법
### 0. 환경 세팅
-추후 작성-
### 1. 파일 구조
```
DataAnalyze   
ㄴdataset (ignored)   
  ㄴflightdata
    ㄴinput
      ㄴT_ONTIME_MARKETING.csv : BTS에서 다운로드 받은 원본 데이터   
      ㄴtailnumTocraft.csv : SFO에서 다운로드 받은 tailnumber에 따른 aircraft 모델 데이터
    ㄴoutput
      ㄴyyyy-mm-dd hh:mm:ss_flight_data_summary.txt : FlightDataGenerator 사용 후 데이터 요약 정보 제공 텍스트 파일
      ㄴyyyy-mm-dd hh:mm:ss_flight_data_summary.txt : FlightDataGenerator 사용 후 생성된 flight data csv 파일
ㄴdatatool   
  ㄴFlightDataGenerator.py : Flight 데이터를 생성하는 파이썬 코드
```
### 2. 사용자 입력 예시
**1) 비행의 기간 입력** : {yyyy-mm-dd hh:mm:ss}와 같은 형태로 입력하기
```
#############    원하는 비행의 기간을 입력해주세요   #############
데이터 시작 일시 : 2023-04-01 00:15:00
데이터 종료 일시 : 2023-05-01 22:25:00
입력 예시: 2023-04-01 00:13:00

✒ 시작 일시: 2023-04-01 00:15:00
✒ 종료 일시: 2023-04-10 00:20:00     
```
**2) 비행기 기종 선택** : 띄어쓰기를 기준으로 원하는 비행기 기종의 번호 입력
```
#############    원하는 비행기의 기종을 입력해주세요   #############
🛫 기종 목록 🛫
{1: 'A200-300', 2: 'A220-100', 3: 'A319-', 4: 'A320-100', 5: 'A320-200', 6: 'A321-100', 7: 'A321-200', 8: 'A330-200', 9: 'A330-300', 10: 'A350-900', 11: 'B717-', 12: 'B737-100', 13: 'B737-200', 14: 'B737-300', 15: 'B737-400', 16: 'B737-700', 17: 'B737-8 Max', 18: 'B737-800', 19: 'B737-9 Max', 20: 'B737-900', 21: 'B757-200', 22: 'B757-300', 23: 'B767-200', 24: 'B767-300', 25: 'B767-400ER', 26: 'B777-200', 27: 'B777-300', 28: 'B787-10', 29: 'B787-8', 30: 'B787-9', 31: 'CRJ-100', 32: 'CRJ-200', 33: 'CRJ-700', 34: 'CRJ-900', 35: 'DC9-30', 36: 'E170-', 37: 'E175-', 38: 'EMB-170', 39: 'EMB-190', 40: 'MD-88'}

✒ 항공사의 번호를 입력해주세요 ex) 1 4 5 : 2 3 4 6 7
```
**3) 공항 선택** : 데이터 개수 기준 상위 몇개의 공항을 볼 것인 지 숫자 입력 후, 원하는 공항의 번호를 띄어쓰기 기준으로 입력
```
#############    원하는 공항의 종류을 입력해주세요   #############
✒ 상위 몇개의 공항을 확인하시겠습니까? : 5

🛫 출발 공항 개수 🛫
     #  count
0  CLT   2427
1  DFW   2194
2  LAX   1202
3  PHX   1048
4  ATL   1009

🛫 도착 공항 개수 🛫
     #  count
0  CLT   2421
1  DFW   2193
2  LAX   1227
3  PHX   1060
4  ATL   1011
🛫 공항 목록 🛫
{1: 'ATL', 2: 'CLT', 3: 'DFW', 4: 'LAX', 5: 'PHX'}

✒ 공항의 번호를 입력해주세요 ex) 1 4 5 : 1 2 3 4 5
```
**4) 항공사 선택** : 원하는 항공사의 번호를 띄어쓰기 기준으로 입력
```
#############    원하는 항공사의 종류을 입력해주세요   #############
🛫 항공사 목록 🛫
{1: 'AA', 2: 'DL'}

✒ 항공사의 번호를 입력해주세요 ex) 1 4 5 : 1
```
**5) 비행의 개수 입력** : 원하는 비행의 수를 입력
```
#############    비행의 수를 입력해주세요   #############
현재 사용할 수 있는 flight의 수 : 1313

✒ 원하는 flight의 개수를 입력하세요 : 200
```
**6) 결과 출력** : 변환을 통해 생성된 flight 데이터의 저장 위치
```
#############    데이터 저장이 완료되었습니다   #############
flight 데이터 요약 정보가 /home/public/yunairline/ASCP/DataAnalyze/dataset/flightdata/output/2023-08-19 10:28:21_flight_data_summary.txt 에 저장되었습니다.
flight 데이터 csv가 /home/public/yunairline/ASCP/DataAnalyze/dataset/flightdata/output/2023-08-19 10:28:21_flight_data.csv 에 저장되었습니다.
```