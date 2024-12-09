#include "Hybrid.h"
#include <algorithm>
#include <cmath>

uint64_t Hybrid::createKey(int userId, int movieId) const
{
  return (static_cast<uint64_t>(userId) << 32) | movieId;
}

double Hybrid::calculateHybridScore(int userId, int movieId) const
{
  uint64_t cacheKey = createKey(userId, movieId);

  // Check cache first
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = hybridScoreCache.find(cacheKey);
    if (it != hybridScoreCache.end())
    {
      return it->second;
    }
  }

  // Get collaborative and content scores
  double collabScore = 0.0;
  auto collabRecs = collaborative.getRecommendations(userId);
  for (const auto &[recMovieId, score] : collabRecs)
  {
    if (recMovieId == movieId)
    {
      collabScore = score;
      break;
    }
  }

  double contentScore = 0.0;
  const auto &userRatings = graph.getUserItems().at(userId);
  double ratingWeight = 0.0;

  // For each movie the user has rated, get its similarity to the target movie
  for (const auto &[ratedMovieId, rating] : userRatings)
  {
    float similarity = content.calculateSimilarity(ratedMovieId, movieId);
    contentScore += similarity * rating;
    ratingWeight += rating;
  }

  if (ratingWeight > 0)
  {
    contentScore /= ratingWeight; // Normalize by total rating weight
  }

  // Get user's PageRank score
  double userRank = pageRank.getPageRank(userId);

  const double BASE_COLLAB_WEIGHT = 0.6;
  const double BASE_CONTENT_WEIGHT = 0.4;

  double adjustedCollabWeight = BASE_COLLAB_WEIGHT * (1.0 + userRank);
  double adjustedContentWeight = BASE_CONTENT_WEIGHT;

  double totalWeight = adjustedCollabWeight + adjustedContentWeight;
  adjustedCollabWeight /= totalWeight;
  adjustedContentWeight /= totalWeight;

  double hybridScore =
      adjustedCollabWeight * collabScore +
      adjustedContentWeight * contentScore;

  // Cache the result
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    hybridScoreCache[cacheKey] = hybridScore;
  }

  return hybridScore;
}

std::vector<std::pair<int, double>> Hybrid::getRecommendations(int userId, size_t n) const
{
  std::vector<std::pair<int, double>> recommendations;
  const auto &items = graph.getItems();
  const auto &userRatings = graph.getUserItems().at(userId);

  // Create set of watched movies
  std::unordered_set<int> watchedMovies;
  for (const auto &[movieId, _] : userRatings)
  {
    watchedMovies.insert(movieId);
  }

  // Get scores for unwatched movies
  for (const auto &[movieId, _] : items)
  {
    if (watchedMovies.count(movieId) == 0)
    {
      double score = calculateHybridScore(userId, movieId);
      recommendations.push_back({movieId, score});
    }
  }

  // Sort by score and get top N
  std::sort(recommendations.begin(), recommendations.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  if (recommendations.size() > n)
  {
    recommendations.resize(n);
  }

  return recommendations;
}
