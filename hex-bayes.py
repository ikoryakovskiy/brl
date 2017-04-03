#!/home/ikoryakovskiy/virtualenv-15.1.0/myVE/bin/python

import os

r = os.system("g++ -O3 -c -fPIC repc.c -o repc.o -fopenmp")
if r == 0:
	os.system("g++ -O3 -shared -Wl,-soname,librepc.so -o librepc.so  repc.o -fopenmp")
	os.system("cython --embed bayes.py")
	os.system("g++ -O3 -I /usr/include/python2.7 -o bayes bayes.c -lpython2.7 -lpthread -lm -lutil -ldl -fopenmp")
	os.system("./bayes")
