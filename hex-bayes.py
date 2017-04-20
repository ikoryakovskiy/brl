import os

r1 = os.system("python setup.py build_ext --inplace")
r2 = os.system("g++ -O3 -c -fPIC repc.c -o repc.o -fopenmp")
if r1 == 0 and r2 == 0:
	os.system("g++ -O3 -shared -Wl,-soname,librepc.so -o librepc.so  repc.o -fopenmp")
	os.system("cython --embed -o cbayes.c bayes.py")
	os.system("g++ -O3 -I /usr/include/python2.7 -o cbayes cbayes.c -lpython2.7 -lpthread -lm -lutil -ldl -fopenmp")
	#os.system("./cbayes")
