## Opta Visualization

1. ASCP/Visualization/ 위치로 이동
2. logs 디렉토리 생성 후 로그파일이 저장된 디렉토리 입력
3. python OptaVisualizer.py {x축 시간 간격(분)} {디렉토리명} 실행
```bash
# x축 시간 간격: 10분, 
# 로그 디렉토리 명: great, job, good
# 출력 파일 명: GJGComparison
$ python OptaVisualizer.py 10 great job good GJGComparison
```
예시 디렉토리 구조
```
ASCP/
└── Visualization/
    ├── OptaVisualizer.py
    ├── utils.py
    ├── logs/
    │   ├── great/
    │   ├── job/
    │   └── good/
    └── README.md
```
