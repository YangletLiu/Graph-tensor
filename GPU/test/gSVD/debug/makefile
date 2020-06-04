#=====================================================
#  CUDA NAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 8.26 2018
#     makefile and src in different dir 
#
#====================================================
include ../../../../make.inc

src_path=..
obj_path=.
src = $(wildcard $(src_path)/*.cpp)
obj = $(patsubst %.cpp,$(obj_path)/%.o,$(notdir $(src)))
CC = gcc
CUNV = nvcc
INCLUDE = -I ../../../include  $(INC)
LIB= -L ../../../lib  $(LIBDIR) $(LINKLIB) -lcugraph
LINK = -link
FLAGS = -DADD_ -std=c++11
target = test
ALL:$(target)
%.o:$(src_path)/%.cpp
	$(CUNV) $< $(INCLUDE) $(LIB)  $(FLAGS) -c $@
$(target):$(obj)
	$(CUNV)  $(LIB) $(FLAGS) $(LINK) -o $@ $(obj)
	@echo 'make file success!'
.PHONY:clean
clean:
	-rm -rf test  $(obj)

