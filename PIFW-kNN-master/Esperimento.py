import sys
import matlab.engine

filename = sys.argv[-1]

# Per lanciare il programma
eng = matlab.engine.start_matlab()

# Per lanciare programma pifwknn
 eng.pifwknn(filename, nargout=0)

#Per lanciare serialized
