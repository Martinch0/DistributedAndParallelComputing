################################################################################
#
# Build script for project
#
################################################################################

# Executable name
EXECUTABLE	:= smb

# CUDA source files (compiled with nvcc)
CUFILES		:= kernel.cu

# CUDA dependency files
CU_DEPS		:= Makefile

# C/C++ source files (compiled with gcc/g++)
CCFILES		:= sweep_optimised.cpp
		   

################################################################################
# Rules and targets

# flag to enable double precision
NVCCFLAGS += -arch sm_20
NVCCFLAGS += -arch=compute_20 -c kernel.cu

# flag to save intermediate files to look at PTX code
# NVCCFLAGS += -keep

# flag to clean up afterwards
# NVCCFLAGS += -keep -clean

# flag to report on use of registers, shared memory, etc
NVCCFLAGS += -Xptxas -v

# Additional compiler flags and LIBs to include
USEGLLIB             := 1
USEGLUT	             := 1
USERENDERCHECKGL     := 1

ROOTDIR = /bham/pd/packages/SL6/x86_64/cuda-4.0.17-sdk/C/common
BINDIR  = ./bin
include $(ROOTDIR)/common.mk
