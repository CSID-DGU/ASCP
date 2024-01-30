from Components import Airport, Aircraft, Hotel
import copy
#[t_ori, t_des, dur, a_ori, a_des, type]
#[  0      1     2     3       4    5  ]

#Aircraft['[0,0,1]'] = [crewnum, layover, quickturn]

def deflect_hard(V_p_list, V_f):
    index_list = []

    i = -1
    for V_p in V_p_list :
        if V_p == [0,0,0,[0],[0],[0]] : break    # pairing_list에 존재하는 모든 pairing을 확인함

        i += 1
        flight_gap = V_f[0] - V_p[1]
        if V_p[1]-V_p[0]>7*24*60: continue       # 총 비행시간 일주일로 설정
        if V_p[4] == V_p[3] : continue           # 완성된 페어링
        if flight_gap < 0 : continue             # 시간의 선후관계 제약
        if V_p[4] != V_f[3] : continue           # 공간 제약
        if V_p[5] != V_f[5] : continue           # 항공기 기종 제약
        if flight_gap < 10*60 :                  # 법적 제약
            if V_p[2] + V_f[2] + flight_gap > 14*60 : continue

        index_list.append(i)
    
    i += 1
    index_list.append(i)
    return index_list


def update_state(V_p_list, V_f, idx) :
    V_p = copy.deepcopy(V_p_list[idx])
    V_f_tmp = copy.deepcopy(V_f)
    
    if V_p == [0,0,0,[0],[0],[0]] :
        V_p = V_f_tmp
    
    else :
        flight_gap = V_f_tmp[0] - V_p[1]
        
        if flight_gap < 10*60 :
            V_p[2] += flight_gap + V_f_tmp[2]    # 휴식시간이 없었다면 V_p_dur에 gap과 c_dur을 더함
        else : V_p[2] = V_f_tmp[2]               # 만약 휴식시간을 가졌다면 dur 초기화
        
        V_p[1] = V_f_tmp[1] # 도착시간
        V_p[4] = V_f_tmp[4] # 도착공항
    
    V_p_list[idx] = V_p


def get_reward(V_p_list, V_f, idx) :
    LAYOVER_TIME = 6*60
    QUICKTURN_TIME = 30
    V_p = V_p_list[idx]
    flight_gap = V_f[0] - V_p[1]
    reward = 0
    
    # Deadhead
    # V_p가 [0,0,0,[0],[0],[0]]이면 새로운 dh가 생기는 것이므로 reward에 추가
    if V_p == [0,0,0,[0],[0],[0]] :
        reward += Airport.get_cost(V_f[4], V_f[3]) *3 # 새 dh 비용 생김
    elif V_p[3] == V_f[4] :
        reward -= Airport.get_cost(V_p[4], V_p[3]) *3 # dh 비용 사라짐
    else :
        reward -= Airport.get_cost(V_p[4], V_p[3]) *3
        reward += Airport.get_cost(V_f[4], V_p[3]) *3 # 원래 dh비용 대신 새 dh비용
    #print('dh: ', reward)
    
    # Layover & Hotel
    if V_p != [0,0,0,[0],[0],[0]] :
        if flight_gap >= LAYOVER_TIME :
            reward += (flight_gap - LAYOVER_TIME) * Aircraft.get_cost(V_f[5])[1]
            V_p_days = V_p[1]//(24*60)
            V_f_days = V_f[0]//(24*60)
            reward += (1 + max((V_f_days - V_p_days) - 1, 0)) * Hotel.get_cost(V_p[4])
        
        # Quickturn
        if flight_gap <= QUICKTURN_TIME :
            reward += Aircraft.get_cost(V_f[5])[2]
        
        # Satis
        if QUICKTURN_TIME < flight_gap < LAYOVER_TIME :
            base = Aircraft.get_cost(V_f[5])[2] # (30min = qt cost) : (6h = 0)
            reward += base // (LAYOVER_TIME - QUICKTURN_TIME) * (LAYOVER_TIME - flight_gap)
        
    return reward