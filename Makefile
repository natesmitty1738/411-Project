CXX = g++
CXXFLAGS = -std=c++17

SRCS = BipartiteGraph.cpp Content.cpp Hybrid.cpp PageRank.cpp Collabrative.cpp
TEST_SRCS = run_tests.cpp

OBJS = $(SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

all: run_tests

run_tests: $(OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o run_tests 