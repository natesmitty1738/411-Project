#ifndef COLLABRATIVE_H
#define COLLABRATIVE_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include "BipartiteGraph.h"
#include "PageRank.h"

class Collaborative
{
private:
  const BipartiteGraph &graph;
  const PageRank &pageRank;

  // Cache for storing similarity scores between user pairs
  mutable std::unordered_map<uint64_t, float> similarityCache;

  // Tracks how many times each cached similarity has been accessed
  mutable std::unordered_map<uint64_t, int> cacheAccessCount;

  // Mutex to ensure thread-safe access to cache structures
  mutable std::mutex cacheMutex;

  // Maximum number of user pairs to keep in similarity cache
  const size_t MAX_CACHE_SIZE = 10000;

  // Minimum number of influential users needed for PageRank-based recommendations
  const size_t MIN_INFLUENTIAL_USERS = 5;

  // Minimum PageRank score to be considered influential
  const double MIN_PAGERANK_SCORE = 0.01;

  // Helper methods
  uint64_t createPairKey(int id1, int id2) const;
  void evictCache() const;

  // New helper method for getting recommendations from influential users
  std::vector<std::pair<int, float>> getInfluentialRecommendations(
      const std::unordered_set<int> &userMovies, size_t n) const;

public:
  explicit Collaborative(const BipartiteGraph &bg, const PageRank &pr)
      : graph(bg), pageRank(pr) {}

  // Calculates similarity between two users using cosine similarity
  float calculateSimilarity(int user1Id, int user2Id) const;

  // Pre-computes similarities between all user pairs in parallel
  void preComputeSimilarities(int numThreads = std::thread::hardware_concurrency());

  // Retrieves cached similarity between two users
  float getCachedSimilarity(int userId1, int userId2) const;

  // Get top N recommendations for a user
  std::vector<std::pair<int, float>> getRecommendations(int userId, size_t n = 5) const;
};

#endif