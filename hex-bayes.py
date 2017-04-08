import os

r = os.system("g++ -g -O0 -c -fPIC repc.c -o repc.o -fopenmp")
if r == 0:
	os.system("g++ -g -O0 -shared -Wl,-soname,librepc.so -o librepc.so  repc.o -fopenmp")
	os.system("cython --embed bayes.py")
	os.system("g++ -g -O0 -I /usr/include/python2.7 -o bayes bayes.c -lpython2.7 -lpthread -lm -lutil -ldl -fopenmp")
	#os.system("./bayes")
