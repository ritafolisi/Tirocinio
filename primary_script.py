import sys
from subprocess import call

data=sys.argv[1]
code=sys.argv[2]

call(["python3", code, data])
