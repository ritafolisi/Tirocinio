import sys
from subprocess import call

data=sys.argv[1]
arg2=sys.argv[2]
arg2=arg2[1:-1]
code=arg2.split(",")

for i in code:
    print("Calling "+ i)
    call(["python3", i, data])
