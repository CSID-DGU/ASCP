# logs 폴더에서 파일 읽어서 파일 이름과 맨 마지막 줄의 best score을 리스트에 저장하기
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class OptaVisualization:
    def __init__(self, grid, dir_name_list, filename):
        # 여러개의 디렉토리를 하나의 그래프로 그리기 위해 디렉토리 이름을 저장하기
        self.dir_name_list = dir_name_list
        self.filename = filename
        self.logs_path_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs/'+dir_name) for dir_name in dir_name_list]
        self.logs_list = [os.listdir(logs_path) for logs_path in self.logs_path_list]
        
        self.dir_best_scores = {}
        # x축의 시간 간격(분)을 입력받아서 그 시간 간격마다의 best score를 저장하기
        self.grid_sec = grid * 60000
        
    def get_best_scores(self, logs, logs_path):
        """
        파일 line예시
        14:20:45.095     LS step (1), time spent (831), score (0hard/-54807478200soft),     best score (0hard/-54807478200soft), accepted/selected move count (1/2), picked move (F39336 {Pairing - 39336 { pair=[F39336] }[0]} <-> F32112 {Pairing - 32112 { pair=[F32112] }[0]}).
        best score를 갱신할 때마다 time spent와 best score를 저장하기
        """
        best_score = None
        best_scores = []
        # logs의 파일들 이름순 정렬
        logs.sort()
        for log in tqdm(logs):
            print(log)
            with open(os.path.join(logs_path, log), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    score = re.findall(r'best score \(\d+hard/-(\d+)soft\)', line)
                    time_spent = re.findall(r'time spent \((\d+)\)', line)
                    if score:
                        try:
                            score = int(score[0])
                        except:
                            continue
                        if best_score is None or score < best_score:
                            best_score = score
                            best_scores.append((int(time_spent[0]), best_score))
        return best_scores
    
    def get_best_scores_by_grid(self, logs, logs_path):
        """
        grid_sec마다의 best score를 저장하기
        """
        best_score_list = self.get_best_scores(logs, logs_path)
        best_score_by_grid = []
        for i in tqdm(range(0, best_score_list[-1][0], self.grid_sec)):
            best_score = None
            for time, score in best_score_list:
                if time <= i:
                    best_score = score
                else:
                    break
            # 시간을 다시 분으로 바꾸기
            i = i // 60000
            best_score_by_grid.append((i, best_score))
        best_score_by_grid.pop(0)
        return best_score_by_grid
    
    def get_dir_best_scores(self):
        # print(self.logs_list)
        # print(self.logs_path_list)
        # print(self.dir_name_list)
        for i, logs in tqdm(enumerate(self.logs_list)):
            best_score_list = self.get_best_scores_by_grid(logs, self.logs_path_list[i])
            self.dir_best_scores[self.dir_name_list[i]] = best_score_list
        return self.dir_best_scores
    
    def filter_max_time(self):
        # 시간이 가장 짧은 그래프의 길이에 맞춰서 다른 그래프의 길이를 맞추기
        min_len = min([len(best_score_list) for best_score_list in self.dir_best_scores.values()])
        for dir_name, best_score_list in tqdm(self.dir_best_scores.items()):
            self.dir_best_scores[dir_name] = best_score_list[:min_len]
        return self.dir_best_scores
    
    def export_graph_table(self):
        # dir_best_scores에 저장된 best score를 그래프로 그리고 표를 저장하기
        """
        표 예시
        |       | dir1 | dir2 | dir3 |
        |-------|------|------|------|
        | time1 |  100 |  200 |  300 |
        | time2 |  200 |  300 |  400 |
        """
        print('exporting graph and table...')
        
        self.get_dir_best_scores()
        self.filter_max_time()
        plt.figure(figsize=(20, 10))
        # 여백 최대한 없애기
        plt.tight_layout()
        # pandas 표 만들기
        table = pd.DataFrame()
        for dir_name, best_score_list in tqdm(self.dir_best_scores.items()):
            plt.plot(range(len(best_score_list)), [score for _, score in best_score_list], label=dir_name)
            # table column = dir_name, row = time, value = score
            table[dir_name] = [score for _, score in best_score_list]
        table.index = [time for time, _ in best_score_list]
        table.to_csv(f'{self.filename}.csv')
        # 글씨 겹치지 않게 크기 조절
        plt.xticks(range(len(best_score_list)), [file for file, _ in best_score_list], rotation=60, fontsize=15)
        plt.legend()
        
        # # 그래프가 꺾이는 부분(점수가 달라지는 부분)에 점 표시 및 점수, 날짜 출력, 모든 그래프에 적용
        # for dir_name, best_score_list in self.dir_best_scores.items():
        #     for i, (time, score) in enumerate(best_score_list):
        #         if i == 0:
        #             continue
        #         # 점수가 달라지는 부분에 점 표시
        #         if best_score_list[i-1][1] != score:
        #             plt.scatter(i, score, c='red')
        #             score_src = f'{score:,}'
        #             plt.text(i, score, f'{score_src}', rotation=30, verticalalignment='bottom', fontsize=7)
        # plt.show()
        plt.savefig(f'{self.filename}.png')
        
        