#include "Content.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <random>

uint64_t Content::createPairKey(int id1, int id2) const
{
  if (id1 > id2)
    std::swap(id1, id2); // Ensure consistent ordering
  return (static_cast<uint64_t>(id1) << 32) | static_cast<uint64_t>(id2);
}

void Content::evictCache() const
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

float Content::calculateSimilarity(int item1Id, int item2Id) const
{
  if (item1Id == item2Id)
    return 1.0f;

  const auto &items = graph.getItems();
  auto item1It = items.find(item1Id);
  auto item2It = items.find(item2Id);

  if (item1It == items.end() || item2It == items.end())
  {
    return 0.0f;
  }

  const auto &item1 = item1It->second;
  const auto &item2 = item2It->second;

  // Calculate genre similarity (Jaccard similarity)
  float genreSimilarity = 0.0f;
  if (!item1.genres.empty() && !item2.genres.empty())
  {
    std::unordered_set<std::string> genres1(item1.genres.begin(), item1.genres.end());
    std::unordered_set<std::string> genres2(item2.genres.begin(), item2.genres.end());

    int intersection = 0;
    for (const auto &genre : genres1)
    {
      if (genres2.count(genre) > 0)
      {
        intersection++;
      }
    }

    genreSimilarity = static_cast<float>(intersection) /
                      (genres1.size() + genres2.size() - intersection);
  }

  // Calculate rating similarity
  float ratingDiff = std::abs(item1.imdb - item2.imdb) / 10.0f; // Normalize to [0,1]
  float ratingSimiliarity = 1.0f - ratingDiff;

  // Calculate year similarity
  float yearDiff = std::abs(static_cast<float>(item1.rating - item2.rating)) / 4.0f; // Normalize to [0,1]
  float yearSimilarity = 1.0f - yearDiff;

  // Calculate length similarity
  float lengthDiff = std::abs(static_cast<float>(item1.length - item2.length)) / 180.0f; // Normalize to [0,1]
  float lengthSimilarity = 1.0f - lengthDiff;

  // Weighted combination
  const float GENRE_WEIGHT = 0.6f;
  const float RATING_WEIGHT = 0.2f;
  const float YEAR_WEIGHT = 0.1f;
  const float LENGTH_WEIGHT = 0.1f;

  float similarity =
      GENRE_WEIGHT * genreSimilarity +
      RATING_WEIGHT * ratingSimiliarity +
      YEAR_WEIGHT * yearSimilarity +
      LENGTH_WEIGHT * lengthSimilarity;

  return similarity;
}

void Content::preComputeSimilarities(int numThreads)
{
  const auto &items = graph.getItems();

  // Create a list of all item pairs to compute
  std::vector<std::pair<int, int>> itemPairs;
  for (const auto &[item1Id, _] : items)
  {
    for (const auto &[item2Id, _] : items)
    {
      if (item1Id < item2Id) // Only compute each pair once
      {
        itemPairs.push_back({item1Id, item2Id});
      }
    }
  }

  // Function to process a chunk of pairs
  auto processPairs = [this](const std::vector<std::pair<int, int>> &pairs, size_t start, size_t end)
  {
    for (size_t i = start; i < end && i < pairs.size(); i++)
    {
      const auto &[item1Id, item2Id] = pairs[i];
      float similarity = calculateSimilarity(item1Id, item2Id);

      if (similarity > 0)
      {
        uint64_t key = createPairKey(item1Id, item2Id);
        std::lock_guard<std::mutex> lock(cacheMutex);
        similarityCache[key] = similarity;
        cacheAccessCount[key] = 1;
      }
    }
  };

  // Split work among threads
  std::vector<std::thread> threads;
  size_t pairsPerThread = (itemPairs.size() + numThreads - 1) / numThreads;

  for (int i = 0; i < numThreads; i++)
  {
    size_t start = i * pairsPerThread;
    size_t end = std::min(start + pairsPerThread, itemPairs.size());
    if (start < end)
    {
      threads.emplace_back(processPairs, std::ref(itemPairs), start, end);
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

float Content::getCachedSimilarity(int itemId1, int itemId2) const
{
  if (itemId1 == itemId2)
    return 1.0f;

  uint64_t key = createPairKey(itemId1, itemId2);
  std::lock_guard<std::mutex> lock(cacheMutex);

  auto it = similarityCache.find(key);
  if (it != similarityCache.end())
  {
    cacheAccessCount[key]++;
    return it->second;
  }

  return 0.0f; // Not found in cache
}

std::vector<std::pair<int, float>> Content::getSimilarItems(int itemId, size_t n) const
{
  const auto &items = graph.getItems();
  if (items.find(itemId) == items.end())
  {
    return {}; // Item not found
  }

  std::vector<std::pair<int, float>> similarities;
  for (const auto &[otherId, _] : items)
  {
    if (otherId != itemId)
    {
      float similarity = getCachedSimilarity(itemId, otherId);
      if (similarity > 0)
      {
        similarities.push_back({otherId, similarity});
      }
    }
  }

  // Sort by similarity score
  std::sort(similarities.begin(), similarities.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

  // Return top N
  if (similarities.size() > n)
  {
    similarities.resize(n);
  }

  return similarities;
}

std::vector<std::pair<int, float>> Content::getRecommendations(int userId, size_t n) const
{
  const auto &users = graph.getUserItems();
  const auto &items = graph.getItems();
  auto userIt = users.find(userId);

  // If user not found or has no ratings, return top rated movies
  if (userIt == users.end() || userIt->second.empty())
  {
    std::vector<std::pair<int, float>> recommendations;
    for (const auto &[movieId, movie] : items)
    {
      recommendations.push_back({movieId, movie.imdb});
    }

    // Sort by IMDB rating
    std::sort(recommendations.begin(), recommendations.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    if (recommendations.size() > n)
    {
      recommendations.resize(n);
    }
    return recommendations;
  }

  // Count genre preferences and calculate average ratings
  std::unordered_map<std::string, std::pair<float, int>> genreStats; // genre -> {total_rating, count}
  std::unordered_set<int> watchedMovies;

  for (const auto &[movieId, rating] : userIt->second)
  {
    auto movieIt = items.find(movieId);
    if (movieIt != items.end())
    {
      watchedMovies.insert(movieId);
      for (const auto &genre : movieIt->second.genres)
      {
        genreStats[genre].first += rating;
        genreStats[genre].second++;
      }
    }
  }

  // Calculate average rating per genre
  std::unordered_map<std::string, float> genrePreferences;
  float maxPreference = 0.0f;
  for (const auto &[genre, stats] : genreStats)
  {
    if (stats.second > 0)
    {
      float avgRating = stats.first / stats.second;
      float preference = avgRating * std::pow(stats.second, 0.5); // Weight by sqrt of count
      genrePreferences[genre] = preference;
      maxPreference = std::max(maxPreference, preference);
    }
  }

  // Normalize preferences
  if (maxPreference > 0)
  {
    for (auto &[_, preference] : genrePreferences)
    {
      preference /= maxPreference;
    }
  }

  // Score all unwatched movies
  std::vector<std::pair<int, float>> recommendations;
  for (const auto &[movieId, movie] : items)
  {
    if (watchedMovies.count(movieId) > 0)
      continue;

    // Calculate genre score
    float genreScore = 0.0f;
    float totalWeight = 0.0f;

    for (const auto &genre : movie.genres)
    {
      auto it = genrePreferences.find(genre);
      if (it != genrePreferences.end())
      {
        genreScore += it->second;
        totalWeight += 1.0f;
      }
    }

    // Normalize genre score
    if (totalWeight > 0)
    {
      genreScore /= totalWeight;
    }

    // Combine genre score with movie quality
    float score = 0.8f * genreScore + 0.2f * (movie.imdb / 10.0f);
    recommendations.push_back({movieId, score});
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
