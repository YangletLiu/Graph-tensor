#=====================================================
#  CUDA MAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 8.23 2018
#     makefile and src in different dir 
#
#====================================================

include ../../make.inc

src_dir=.
bin_dir=../bin
lib_dir=../lib
inc_dir=../include
src :=  $(wildcard $(src_dir)/gFT/*.cu)\
	$(wildcard $(src_dir)/gIFT/*.cu) \
	$(wildcard $(src_dir)/gFilter/*.cu) \
	$(wildcard $(src_dir)/gShift/*.cu) \
	$(wildcard $(src_dir)/gConvolution/*.cu) \
	$(wildcard $(src_dir)/gProduct/*.cu) \
	$(wildcard $(src_dir)/gSVD/*.cu) \
	$(wildcard $(src_dir)/gQR/*.cu) \
	$(wildcard $(src_dir)/*.cu)
 
obj  += $(patsubst %.cu,$(bin_dir)/%.o,$(notdir $(src)))

AR = $(ARCH)
#TARGET =$(obj)
TARGET = $(lib_dir)/libcugraph.a
#CUNV = nvcc
#INCLUDE = -I$(inc_dir) -I/usr/loacl/cuda/include 
#FLAGS = -DADD_  -std=c++11
#LINK = -lcusolver -lcusparse -lcublas -lcudart -lcufft -lcurand

CUNV    = $(NVCC)
INCLUDE = -I$(inc_dir) $(INC)
LIB     = $(LIBDIR)
FLAGS   = $(NVCCFLAGS) $(CXXFLAGS)
#CODE_FLAGS = $(GENCODE_FLAGS)
LINK    = $(LINKLIB)

ALL:$(obj)
$(bin_dir)/%.o:$(src_dir)/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gFT/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gIFT/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gFilter/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gShift/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gConvolution/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gProduct/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gSVD/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<
$(bin_dir)/%.o:$(src_dir)/gQR/%.cu
	$(CUNV) $(INCLUDE) $(LIB) $(CODE_FLAGS) $(FLAGS) -c -o $(bin_dir)/$@  $<


.PHONY:clean
clean:
	-rm -rf $(obj) $(TARGET)
