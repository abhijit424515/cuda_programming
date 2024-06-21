.DEFAULT_GOAL := start
TARGET=runme

start:
	nvcc -arch=sm_86 $(path) -o $(TARGET)

clean:
	rm -f $(TARGET)