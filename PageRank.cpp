#include "PageRank.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

PageRank::PageRank(const BipartiteGraph &bg) : graph(bg)
{
  calculatePageRanks();
}

void PageRank::initializeRanks() const
{
  const auto &users = graph.getUserItems();
  if (users.empty())
    return;

  double initialRank = 1.0 / users.size();
  ranks.clear();
  for (const auto &[userId, _] : users)
  {
    ranks[userId] = initialRank;
  }
}

void PageRank::normalizeRanks(std::unordered_map<int, double> &ranks) const
{
  double sum = 0.0;
  for (const auto &[_, rank] : ranks)
  {
    sum += rank;
  }

  if (sum > 0)
  {
    for (auto &[_, rank] : ranks)
    {
      rank = std::max(MIN_RANK, rank / sum);
    }
  }
}

double PageRank::calculateActivityScore(size_t numRatings, size_t maxRatings) const
{
  if (maxRatings == 0)
    return 1.0;

  double activityRatio = static_cast<double>(numRatings) / maxRatings;
  if (activityRatio >= CORE_ACTIVITY_THRESHOLD)
  {
    return ACTIVITY_BOOST; // Core users get a significant boost
  }
  else
  {
    // Sigmoid function for smooth transition
    return 1.0 + (ACTIVITY_BOOST - 1.0) / (1.0 + std::exp(-10 * (activityRatio - CORE_ACTIVITY_THRESHOLD)));
  }
}

void PageRank::calculatePageRanks() const
{
  const auto &users = graph.getUserItems();
  const auto &items = graph.getItems();

  if (users.empty() || items.empty())
  {
    return;
  }

  // Initialize ranks
  initializeRanks();

  // Find maximum number of ratings by any user
  size_t maxRatings = 0;
  for (const auto &[_, userRatings] : users)
  {
    maxRatings = std::max(maxRatings, userRatings.size());
  }

  // Iterative PageRank calculation
  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    std::unordered_map<int, double> newRanks;
    double totalDiff = 0.0;

    // Calculate new rank for each user
    for (const auto &[userId, userRatings] : users)
    {
      // Get activity score based on number of ratings
      double activityScore = calculateActivityScore(userRatings.size(), maxRatings);

      // Initialize with damping factor
      double newRank = (1.0 - DAMPING) / users.size();

      // Create set of movies rated by this user
      std::unordered_set<int> userMovies;
      for (const auto &[movieId, _] : userRatings)
      {
        if (items.find(movieId) != items.end())
        {
          userMovies.insert(movieId);
        }
      }

      // Calculate contribution from other users through shared movies
      for (const auto &[otherId, otherRatings] : users)
      {
        if (otherId == userId)
          continue;

        // Count shared movies
        int sharedMovies = 0;
        for (const auto &[movieId, _] : otherRatings)
        {
          if (userMovies.count(movieId) > 0)
          {
            sharedMovies++;
          }
        }

        if (sharedMovies > 0)
        {
          double contribution = ranks[otherId] * sharedMovies / otherRatings.size();
          newRank += DAMPING * activityScore * contribution;
        }
      }

      newRanks[userId] = newRank;
      totalDiff += std::abs(newRank - ranks[userId]);
    }

    // Normalize new ranks
    normalizeRanks(newRanks);

    // Update ranks
    ranks = std::move(newRanks);

    // Check for convergence
    if (totalDiff < CONVERGENCE_THRESHOLD)
    {
      break;
    }
  }
}

double PageRank::getPageRank(int userId) const
{
  auto it = ranks.find(userId);
  return it != ranks.end() ? it->second : MIN_RANK;
}
