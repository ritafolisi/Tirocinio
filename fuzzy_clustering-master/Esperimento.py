import sys
import matlab.engine

filename = sys.argv[-1]

# Per lanciare il programma
eng = matlab.engine.start_matlab()
# eng.pifwknn(nargout=0)

eng.test(nargout=0)
