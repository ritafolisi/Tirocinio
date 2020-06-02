import sys
import matlab.engine

filename = sys.argv[-1]

# Per lanciare il programma
eng = matlab.engine.start_matlab()
# eng.pifwknn(nargout=0)

eng.pifwknn(filename, nargout=0)
