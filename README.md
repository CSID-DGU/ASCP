# ASCP
Airline Scheduling Crew Pairing

# GitHub Role
다음과 같은 Convetion을 따릅니다.

## Commit Convention
-   feat : 새로운 기능 추가
-   fix : 버그 수정
-   docs : 문서 수정
-   style : 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우
-   refactor: 코드 리펙토링
-   test: 테스트 코드, 리펙토링 테스트 코드 추가
-   chore : 빌드 업무 수정, 패키지 매니저 수정

## 💡 PR Convetion

| 아이콘 | 코드                       | 설명                     |
| ------ | -------------------------- | ------------------------ |
| 🎨     | :art                       | 코드의 구조/형태 개선    |
| ⚡️    | :zap                       | 성능 개선                |
| 🔥     | :fire                      | 코드/파일 삭제           |
| 🐛     | :bug                       | 버그 수정                |
| 🚑     | :ambulance                 | 긴급 수정                |
| ✨     | :sparkles                  | 새 기능                  |
| 💄     | :lipstick                  | UI/스타일 파일 추가/수정 |
| ⏪     | :rewind                    | 변경 내용 되돌리기       |
| 🔀     | :twisted_rightwards_arrows | 브랜치 합병              |
| 💡     | :bulb                      | 주석 추가/수정           |
| 🗃      | :card_file_box             | 데이버베이스 관련 수정   |

## Intro
Optaplanner를 서버에서 실행하는 방법입니다.
<pre>
./gradlew build
cp crew-pairing.jar ~/ASCP/PairingCreater/
java -jar crew-pairing.jar data/ crewpairing/ 500 input_500.xlsx
java -jar crew-pairing.jar data/ crewpairing/ 500 input_500.xlsx output.xlsx
</pre>
- 프로젝트 루트 디렉토리에서 ./gradlew build 실행하면 build/libs에 crew-pairing.jar 파일 생성됨
- crew-pairing.jar을 프로젝트 루트 디렉토리로 이동
- data/crewpairing/ 디렉토리에 output 폴더 추가
- 나머지 두 명령어를 통해 Optaplanner 실행
- 아래는 initial set (output.xlsx)이 존재하는 경우 사용. data/crewpairing/output/ 에 initial set 파일 배치

RL을 서버 및 로컬에서 실행하는 방법입니다.
- 프로젝트 루트 디렉토리에 dataset 폴더 추가 후, input 파일 배치
- REINFORCE.py 실행

## Lisence
