#=====================================================
#  CUDA MAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 8.26 2018
#     makefile and src in different dir 
#
#====================================================
src_path=..
obj_path=.
src = $(wildcard $(src_path)/*.cpp)
obj = $(patsubst %.cpp,$(obj_path)/%.o,$(notdir $(src)))
CC = gcc
CUNV = nvcc
INCLUDE = -I ../../../include
LIB= -L ../../../lib -lcugraph -lcublas -lcusparse -lcudart -lcufft -lcusolver
LINK = -link
target = test
ALL:$(target)
%.o:$(src_path)/%.cpp
	$(CUNV) $< $(INCLUDE) $(LIB) -c $@
$(target):$(obj)
	$(CUNV)  $(LIB) $(LINK) -o $@ $(obj)
	@echo 'make file success!'
.PHONY:clean
clean:
	-rm -rf test  $(obj)

