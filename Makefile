.PHONY: all clean

CXX=clang++
PYTHON?=python3

CFLAGS=-std=c++11 -g -Wall -Wextra -O3 `pkg-config opencv --cflags`
OPT= -msse -msse2 -msse3 -ffast-math
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_video
LDFLAGS=

NUMPY_INCLUDE_PATH=$(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')
PY_CFLAGS =  $(shell pkg-config $(PYTHON) --cflags) -I$(NUMPY_INCLUDE_PATH)
PY_LDFLAGS = $(shell pkg-config $(PYTHON) --libs)

SRCDIR=src
OBJDIR=obj
OBJS=$(addprefix $(OBJDIR)/,eye.o util.o ellipse.o tracking.o)

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) -c -fPIC $(CFLAGS) $(OPT) $< -o $@

all : eye iris_track.so

$(OBJS) : | $(OBJDIR)
$(OBJDIR) :
	test -d $(OBJDIR) || mkdir $(OBJDIR)

eye : $(OBJS)
	$(CXX) $(CFLAGS) $(OPT) $(LIBS) $(LDFLAGS) $^ -o $@

iris_track.so: $(OBJS) $(SRCDIR)/iris_track.cpp
	$(CXX) -fPIC -shared $(OPT) $(LIBS) \
		$(CFLAGS) $(LDFLAGS) \
		$(PY_CFLAGS) $(PY_LDFLAGS) \
		$^ -o iris_track.so

clean :
	-rm -r $(OBJDIR)
