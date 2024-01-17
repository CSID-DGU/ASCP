import functions as fn
# [t_ori, t_des, dur, a_ori, a_des, type]
# [  0      1     2     3       4    5  ]
# Pairing List -> 유효성 검사를 우선적으로 거친 페어링 리스트

# 문제 정의
GoodPairing = [0,0,0,[0],[0],[0]]
Pairing_list = [[3,0,0,[0],[0],[0]],[3,0,0,[0],[0],[0]]]


#페어링 리스트에서 이상적인 페어링과 유사한 페어링을 찾는 함수
def findSimiliar(GoodPairing, Pairing_list, index_list):
    min_destTime = float('inf') # 도착 시간 차이의 최솟값
    min_originTime = float('inf') # 출발 시간 차이의 최솟값
    min_index = index_list[-1]

    # 가능 페어링이 존재하는지 확인
    for i in index_list:
        # 출발공항과 도착공항이 같아야함
        if GoodPairing[3] == Pairing_list[i][3] and GoodPairing[4]==Pairing_list[i][4]:
            dest_differnce =abs(Pairing_list[i][1]-GoodPairing[1]) # 도착시간 차이
            origin_differnce = abs(Pairing_list[i][0] - GoodPairing[0])  # 출발 시간 차이
            # 페어링 리스트에서 도착 시간 차이의 절댓 값이 가장 작은 페어링을 찾음
            if min_destTime - dest_differnce > 0: # 현재 페어링에서 도착시간 차이가 min보다 작으면
                min_destTime = dest_differnce
                min_originTime = origin_differnce
                min_index = i

            elif min_destTime - dest_differnce == 0: # 도착시간 차이가 같을 경우
                if min_originTime - origin_differnce > 0: # 출발 시간 차이가 0보다 크면
                    min_originTime = origin_differnce
                    min_index = i

    return min_index

#이상적인 페어링과 flight가 hard 제약을 어기는지 확인하는 함수
def checkHard(goodPairing, flight):

    flight_gap = flight[0] - goodPairing[1]

    if flight_gap < 0: return False  # 시간의 선후관계 제약
    if goodPairing[4] != flight[3]: return False  # 공간 제약
    if goodPairing[5] != flight[5]: return False  # 항공기 기종 제약
    if flight_gap < 10 * 60:  # 법적 제약
        if goodPairing[2] + flight[2] + flight_gap > 14 * 60: return  False

    return True

