CXX = g++
CXXFLAGS = -std=c++14 -O2 -Iinclude -Wall -Wextra

TARGET = msar_reproduce
TEST_TARGET = run_tests

all: $(TARGET) $(TEST_TARGET)

$(TARGET): src/main.cpp
	$(CXX) $(CXXFLAGS) src/main.cpp -o $(TARGET)

$(TEST_TARGET): tests/test_pipeline.cpp
	$(CXX) $(CXXFLAGS) tests/test_pipeline.cpp -o $(TEST_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

plots:
	python scripts/plot_results.py

bert-train:
	python scripts/bert/train_cv.py

bert-test:
	python scripts/bert/predict_test.py

clean:
	rm -f $(TARGET) $(TEST_TARGET) *.exe
