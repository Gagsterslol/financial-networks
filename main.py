sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import mst, analysis

def main():
    trees = mst.rolling_window_mst()

    properties = analysis.trees_to_properties(trees)


                  
