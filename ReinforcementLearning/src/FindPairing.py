import functions as fn
# [t_ori, t_des, dur, a_ori, a_des, type]
# [  0      1     2     3       4    5  ]
# Pairing List -> 유효성 검사를 우선적으로 거친 페어링 리스트

# 문제 정의
GoodPairing = [0,0,0,[0],[0],[0]]
Pairing_list = [[3,0,0,[0],[0],[0]],[3,0,0,[0],[0],[0]]]


#DK_Algorithm 함수
def DK_Algorithm(GoodPairing, Pairing_list, index_list):
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

answer= DK_Algorithm(GoodPairing, Pairing_list, [0,1])

print(answer)