#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>

namespace TestUtils
{

  // Helper function to calculate overlap between two recommendation lists
  template <typename T>
  inline double calculateRecommendationOverlap(const std::vector<std::pair<int, T>> &rec1,
                                               const std::vector<std::pair<int, T>> &rec2)
  {
    int common = 0;
    for (const auto &[id1, _] : rec1)
    {
      for (const auto &[id2, __] : rec2)
      {
        if (id1 == id2)
          common++;
      }
    }
    return static_cast<double>(common) / std::min(rec1.size(), rec2.size());
  }

  // Utility functions for generating test data
  inline std::vector<std::string> generateRandomGenres(int count, std::mt19937 &rng)
  {
    std::vector<std::string> allGenres = {"Action", "Adventure", "Comedy", "Drama", "Horror",
                                          "Romance", "Sci-Fi", "Thriller", "Family", "Fantasy"};
    std::vector<std::string> selectedGenres;
    std::shuffle(allGenres.begin(), allGenres.end(), rng);
    for (int i = 0; i < std::min(count, (int)allGenres.size()); i++)
    {
      selectedGenres.push_back(allGenres[i]);
    }
    return selectedGenres;
  }

  inline std::vector<std::pair<int, float>> generateRandomRatings(int movieCount, int ratingCount, std::mt19937 &rng)
  {
    std::vector<std::pair<int, float>> ratings;
    std::vector<int> movieIds(movieCount);
    for (int i = 0; i < movieCount; i++)
    {
      movieIds[i] = i + 1;
    }

    std::shuffle(movieIds.begin(), movieIds.end(), rng);
    std::uniform_real_distribution<float> rating_dist(1.0, 5.0);

    for (int i = 0; i < std::min(ratingCount, movieCount); i++)
    {
      ratings.push_back({movieIds[i], rating_dist(rng)});
    }

    return ratings;
  }

}