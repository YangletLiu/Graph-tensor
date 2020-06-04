TAR = ./GPU/lib/libcugraph.a
all:
	make -C ./GPU/operations
	ar rs $(TAR) $(wildcard ./GPU/bin/*.o)
	@echo "make file success"

.PHONY:clean
clean:
	rm -rf $(TAR) $(wildcard ./GPU/bin/*.o)
