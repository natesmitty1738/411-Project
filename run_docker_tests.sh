# Build Docker image
docker build -t recommender-tests .

# Run tests
docker run --rm recommender-tests 