FROM gcc:latest

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
COPY . .

# Force clean build
RUN rm -rf *.o run_tests

# Build the project
RUN make clean && make

# Create test data directory
RUN mkdir -p /app/test_data

# Set up test environment
ENV TEST_DATA_DIR=/app/test_data

# Command to run tests
CMD ["./run_tests"] 