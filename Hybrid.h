#ifndef HYBRID_H
#define HYBRID_H

#include "BipartiteGraph.h"
#include "Collabrative.h"
#include "Content.h"
#include "PageRank.h"
#include <unordered_map>
#include <vector>
#include <mutex>

class Hybrid
{
private:
  const BipartiteGraph &graph;
  Collaborative &collaborative;
  Content &content;
  PageRank pageRank;

  // Cache for hybrid scores
  mutable std::unordered_map<uint64_t, double> hybridScoreCache;
  mutable std::mutex cacheMutex;

  // Helper methods
  uint64_t createKey(int userId, int movieId) const;

public:
  Hybrid(const BipartiteGraph &bg, Collaborative &collab, Content &cont)
      : graph(bg), collaborative(collab), content(cont), pageRank(bg)
  {
  }

  // Get weighted hybrid recommendations for a user
  std::vector<std::pair<int, double>> getRecommendations(int userId, size_t n = 10) const;

  // Calculate hybrid score incorporating PageRank
  double calculateHybridScore(int userId, int movieId) const;

  // Add this method to get a user's PageRank
  double getUserPageRank(int userId) const
  {
    return pageRank.getPageRank(userId);
  }
};

#endif