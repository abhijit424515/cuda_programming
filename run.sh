compile() {
	nvcc -arch=sm_86 $@ -o main
}

case $1 in
	make) compile ${@:2} ;;
	clean) rm main ;;
	*) echo "Compilation failed" ;;
esac