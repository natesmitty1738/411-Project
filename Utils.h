#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

class Utils
{
public:
  // Calculates cosine similarity between two vectors of ratings
  // Each vector contains pairs of (id, rating) where:
  // - For user similarity: id = movieId, rating = user's rating for that movie
  // - For movie similarity: id = userId, rating = user's rating for that movie
  //
  // Cosine similarity measures the cosine of the angle between two vectors:
  // cos() = (A·B)/(||A||·||B||)
  // Where: A·B is dot product, ||A|| and ||B|| are vector magnitudes
  // Returns value between -1 and 1:
  // 1: Vectors are identical
  // 0: Vectors are perpendicular (completely different)
  // -1: Vectors are opposite
  static float cosineSimilarity(const std::vector<std::pair<int, float>> &vec1,
                                const std::vector<std::pair<int, float>> &vec2)
  {
    // Initialize accumulators for:
    // dotProduct: sum of rating1 * rating2 for shared items
    // norm1, norm2: sum of squared ratings (for magnitude calculation)
    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    // Convert first vector to hash map for O(1) lookup
    // Key: id (movie or user), Value: rating
    // This optimization helps when vectors are sparse
    // (i.e., users rate only a small subset of all movies)
    std::unordered_map<int, float> map1;
    for (const auto &[id, rating] : vec1)
    {
      map1[id] = rating;        // Store rating for O(1) lookup
      norm1 += rating * rating; // Add to squared magnitude
    }

    // Process second vector and calculate dot product
    // We can calculate norm2 in the same loop for efficiency
    for (const auto &[id, rating] : vec2)
    {
      auto it = map1.find(id); // O(1) lookup in hash map
      if (it != map1.end())    // If ID exists in both vectors
      {
        // Multiply ratings for same ID and add to dot product
        dotProduct += it->second * rating;
      }
      norm2 += rating * rating; // Add to squared magnitude
    }

    // Only calculate similarity if both vectors have non-zero magnitude
    // This avoids division by zero and handles empty vectors
    if (norm1 > 0.0f && norm2 > 0.0f)
    {
      // Cosine similarity formula:
      // dot_product / (magnitude1 * magnitude2)
      // sqrt of sum of squares gives magnitude
      return dotProduct / (std::sqrt(norm1) * std::sqrt(norm2));
    }
    return 0.0f; // Return 0 for zero vectors (no similarity)
  }

  // Calculates Jaccard similarity between two sets of strings
  // Jaccard similarity = |A ∩ B| / |A ∪ B|
  // Where A and B are sets, ∩ is intersection, ∪ is union
  // Returns value between 0 and 1:
  // 1: Sets are identical
  // 0: Sets have no elements in common
  // Used for comparing sets of genres, tags, or other categorical data
  static float jaccardSimilarity(const std::vector<std::string> &set1,
                                 const std::vector<std::string> &set2)
  {
    // Convert vectors to sets for O(1) lookup
    std::unordered_set<std::string> s1(set1.begin(), set1.end());
    std::unordered_set<std::string> s2(set2.begin(), set2.end());

    // Calculate intersection size (number of common elements)
    int intersection = 0;
    for (const auto &item : s1)
    {
      if (s2.count(item) > 0) // O(1) lookup in hash set
      {
        intersection++;
      }
    }

    // Calculate union size using inclusion-exclusion principle
    // |A ∪ B| = |A| + |B| - |A ∩ B|
    int union_size = s1.size() + s2.size() - intersection;

    // Return Jaccard similarity
    // Handle empty sets case to avoid division by zero
    return union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f;
  }
};

#endif