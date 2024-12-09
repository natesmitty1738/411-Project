#include "Collabrative.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <thread>

// Creates a unique 64-bit key for caching similarity between two users
// Ensures consistent key generation regardless of parameter order
uint64_t Collaborative::createPairKey(int id1, int id2) const
{
  if (id1 > id2)
    std::swap(id1, id2); // Ensure consistent ordering
  return (static_cast<uint64_t>(id1) << 32) | static_cast<uint64_t>(id2);
}

// Implements least-frequently-used cache eviction policy
void Collaborative::evictCache() const
{
  // Create vector of pairs (key, access count)
  std::vector<std::pair<uint64_t, int>> cacheStats;
  for (const auto &[key, count] : cacheAccessCount)
  {
    cacheStats.push_back({key, count});
  }

  // Sort by access count (least accessed first)
  std::sort(cacheStats.begin(), cacheStats.end(),
            [](const auto &a, const auto &b)
            { return a.second < b.second; });

  // Remove least accessed entries until we're under the limit
  size_t numToRemove = similarityCache.size() - MAX_CACHE_SIZE / 2; // Remove half
  for (size_t i = 0; i < numToRemove && i < cacheStats.size(); i++)
  {
    uint64_t key = cacheStats[i].first;
    similarityCache.erase(key);
    cacheAccessCount.erase(key);
  }
}

// Calculates cosine similarity between two users based on their movie ratings
float Collaborative::calculateSimilarity(int user1Id, int user2Id) const
{
  if (user1Id == user2Id)
    return 1.0f;

  const auto &users = graph.getUserItems();
  auto user1It = users.find(user1Id);
  auto user2It = users.find(user2Id);

  // If either user doesn't exist, return 0
  if (user1It == users.end() || user2It == users.end())
  {
    return 0.0f;
  }

  const auto &ratings1 = user1It->second;
  const auto &ratings2 = user2It->second;

  // If either user has no ratings, return 0
  if (ratings1.empty() || ratings2.empty())
  {
    return 0.0f;
  }

  // Create maps for O(1) lookup
  std::unordered_map<int, float> user1Ratings;
  for (const auto &[movieId, rating] : ratings1)
  {
    user1Ratings[movieId] = rating;
  }

  // Calculate dot product and magnitudes
  double dotProduct = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  int commonMovies = 0;

  // Calculate dot product and norm2
  for (const auto &[movieId, rating2] : ratings2)
  {
    auto it = user1Ratings.find(movieId);
    if (it != user1Ratings.end())
    {
      dotProduct += it->second * rating2;
      commonMovies++;
    }
    norm2 += rating2 * rating2;
  }

  // Calculate norm1
  for (const auto &[_, rating1] : ratings1)
  {
    norm1 += rating1 * rating1;
  }

  // Avoid division by zero and require at least 1 movie in common
  if (norm1 == 0.0 || norm2 == 0.0 || commonMovies < 1)
  {
    return 0.0f;
  }

  return static_cast<float>(dotProduct / (std::sqrt(norm1) * std::sqrt(norm2)));
}

// Pre-computes similarities between all users using multiple threads
void Collaborative::preComputeSimilarities(int numThreads)
{
  const auto &users = graph.getUserItems();

  // Create a list of all user pairs to compute
  std::vector<std::pair<int, int>> userPairs;
  for (const auto &[user1Id, _] : users)
  {
    for (const auto &[user2Id, _] : users)
    {
      if (user1Id < user2Id)
      { // Only compute each pair once
        userPairs.push_back({user1Id, user2Id});
      }
    }
  }

  // Function to process a chunk of pairs
  auto processPairs = [this](const std::vector<std::pair<int, int>> &pairs, size_t start, size_t end)
  {
    for (size_t i = start; i < end && i < pairs.size(); i++)
    {
      const auto &[user1Id, user2Id] = pairs[i];
      float similarity = calculateSimilarity(user1Id, user2Id);

      if (similarity > 0)
      {
        uint64_t key = createPairKey(user1Id, user2Id);
        std::lock_guard<std::mutex> lock(cacheMutex);
        similarityCache[key] = similarity;
        cacheAccessCount[key] = 1;
      }
    }
  };

  // Split work among threads
  std::vector<std::thread> threads;
  size_t pairsPerThread = (userPairs.size() + numThreads - 1) / numThreads;

  for (int i = 0; i < numThreads; i++)
  {
    size_t start = i * pairsPerThread;
    size_t end = std::min(start + pairsPerThread, userPairs.size());
    if (start < end)
    {
      threads.emplace_back(processPairs, std::ref(userPairs), start, end);
    }
  }

  // Wait for all threads to complete
  for (auto &thread : threads)
  {
    thread.join();
  }

  // Evict cache if necessary
  if (similarityCache.size() > MAX_CACHE_SIZE)
  {
    evictCache();
  }
}

// Retrieves cached similarity between two users
float Collaborative::getCachedSimilarity(int userId1, int userId2) const
{
  if (userId1 == userId2)
    return 1.0f;

  uint64_t key = createPairKey(userId1, userId2);
  std::lock_guard<std::mutex> lock(cacheMutex);

  auto it = similarityCache.find(key);
  if (it != similarityCache.end())
  {
    cacheAccessCount[key]++;
    return it->second;
  }

  return 0.0f; // Not found in cache
}

std::vector<std::pair<int, float>> Collaborative::getInfluentialRecommendations(
    const std::unordered_set<int> &userMovies, size_t n) const
{

  const auto &users = graph.getUserItems();
  const auto &items = graph.getItems();

  // Get users sorted by PageRank
  std::vector<std::pair<int, double>> usersByRank;
  for (const auto &[userId, _] : users)
  {
    double rank = pageRank.getPageRank(userId);
    if (rank >= MIN_PAGERANK_SCORE)
    {
      usersByRank.push_back({userId, rank});
    }
  }

  // Sort by PageRank score descending
  std::sort(usersByRank.begin(), usersByRank.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  // If we don't have enough influential users, return empty
  if (usersByRank.size() < MIN_INFLUENTIAL_USERS)
  {
    return {};
  }

  // Collect weighted recommendations from influential users
  std::unordered_map<int, std::pair<float, float>> weightedRecs; // {movieId: {weighted_sum, weight_sum}}

  for (const auto &[userId, rank] : usersByRank)
  {
    auto userIt = users.find(userId);
    if (userIt == users.end())
      continue; // Skip if user not found

    const auto &userRatings = userIt->second;
    float weight = static_cast<float>(rank);

    for (const auto &[movieId, rating] : userRatings)
    {
      // Skip movies that don't exist or user has already rated
      if (items.find(movieId) == items.end() || userMovies.count(movieId) > 0)
      {
        continue;
      }
      weightedRecs[movieId].first += rating * weight;
      weightedRecs[movieId].second += weight;
    }
  }

  // Convert to recommendations
  std::vector<std::pair<int, float>> recommendations;
  for (const auto &[movieId, weights] : weightedRecs)
  {
    if (weights.second > 0)
    {
      float score = weights.first / weights.second;
      // Blend with movie quality
      auto movieIt = items.find(movieId);
      if (movieIt != items.end())
      {
        score = 0.7f * score + 0.3f * movieIt->second.imdb;
      }
      recommendations.push_back({movieId, score});
    }
  }

  // Sort by score
  std::sort(recommendations.begin(), recommendations.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  // Return top N
  if (recommendations.size() > n)
  {
    recommendations.resize(n);
  }

  return recommendations;
}

std::vector<std::pair<int, float>> Collaborative::getRecommendations(int userId, size_t n) const
{
  const auto &users = graph.getUserItems();
  const auto &items = graph.getItems();
  auto userIt = users.find(userId);

  // Get user's current movies
  std::unordered_set<int> userMovies;
  if (userIt != users.end())
  {
    for (const auto &[movieId, _] : userIt->second)
    {
      if (items.find(movieId) != items.end())
      { // Only include existing movies
        userMovies.insert(movieId);
      }
    }
  }

  // Handle new users or users with no ratings
  if (userIt == users.end() || userIt->second.empty())
  {
    auto recommendations = getInfluentialRecommendations(userMovies, n);
    if (!recommendations.empty())
    {
      return recommendations;
    }

    // Fall back to movie quality
    recommendations.reserve(items.size());
    for (const auto &[movieId, item] : items)
    {
      if (userMovies.count(movieId) == 0)
      {
        recommendations.push_back({movieId, item.imdb});
      }
    }

    std::sort(recommendations.begin(), recommendations.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    if (recommendations.size() > n)
    {
      recommendations.resize(n);
    }
    return recommendations;
  }

  // Calculate weighted scores for all unwatched movies
  std::unordered_map<int, std::pair<float, float>> weightedScores; // movieId -> {score_sum, weight_sum}

  // Find similar users
  std::vector<std::pair<int, float>> similarUsers;
  for (const auto &[otherId, _] : users)
  {
    if (otherId == userId)
      continue;

    float similarity = getCachedSimilarity(userId, otherId);
    if (similarity > 0)
    {
      similarUsers.push_back({otherId, similarity});
    }
  }

  // Sort by similarity
  std::sort(similarUsers.begin(), similarUsers.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  // Take top K similar users
  const size_t K = 10;
  if (similarUsers.size() > K)
  {
    similarUsers.resize(K);
  }

  // Get recommendations from similar users
  for (const auto &[otherId, similarity] : similarUsers)
  {
    auto otherIt = users.find(otherId);
    if (otherIt == users.end())
      continue;

    float weight = similarity * pageRank.getPageRank(otherId); // Weight by similarity and PageRank

    for (const auto &[movieId, rating] : otherIt->second)
    {
      if (items.find(movieId) == items.end() || userMovies.count(movieId) > 0)
      {
        continue;
      }
      weightedScores[movieId].first += rating * weight;
      weightedScores[movieId].second += weight;
    }
  }

  // Convert weighted scores to recommendations
  std::vector<std::pair<int, float>> recommendations;
  for (const auto &[movieId, weights] : weightedScores)
  {
    if (weights.second > 0)
    {
      float score = weights.first / weights.second;
      // Blend with movie quality
      auto movieIt = items.find(movieId);
      if (movieIt != items.end())
      {
        score = 0.8f * score + 0.2f * movieIt->second.imdb;
      }
      recommendations.push_back({movieId, score});
    }
  }

  // Sort by score
  std::sort(recommendations.begin(), recommendations.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  if (recommendations.size() > n)
  {
    recommendations.resize(n);
  }

  return recommendations;
}
