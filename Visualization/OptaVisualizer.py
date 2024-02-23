import sys
from utils import OptaVisualization

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python OptaVisualization.py <grid> filename')
        sys.exit(1)
    grid = int(sys.argv[1])
    # dir_name_list 에 나머지 인자들을 넣어주기
    dir_name_list = sys.argv[2:-1]
    filename = sys.argv[-1]
    Op = OptaVisualization(grid, dir_name_list, filename)
    Op.export_graph_table()