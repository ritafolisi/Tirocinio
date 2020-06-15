import sys
import matlab.engine

filename = sys.argv[1]

# Per lanciare il programma
eng = matlab.engine.start_matlab()

eng.esperimento_main(nargout=0)
