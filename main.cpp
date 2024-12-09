#include "BipartiteGraph.h"
#include "Collabrative.h"
#include "Content.h"
#include "Hybrid.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

// Utility functions for loading data
void loadMovies(BipartiteGraph &bg, const string &filename)
{
  ifstream file(filename);
  string line;

  while (getline(file, line))
  {
    stringstream ss(line);
    string item;
    vector<string> row;

    while (getline(ss, item, ','))
    {
      row.push_back(item);
    }

    if (row.size() >= 5)
    {
      int id = stoi(row[0]);
      vector<string> genres;
      stringstream genreSS(row[2]);
      string genre;
      while (getline(genreSS, genre, '|'))
      {
        genres.push_back(genre);
      }
      float length = stof(row[3]);
      float rating = stof(row[4]);
      int year = stoi(row[1]);

      bg.addItem(id, genres, length, rating, year);
    }
  }
}

void loadRatings(BipartiteGraph &bg, const string &filename)
{
  ifstream file(filename);
  string line;

  // Skip header
  getline(file, line);

  // Keep track of user ratings
  unordered_map<int, vector<pair<int, float>>> userRatings;

  while (getline(file, line))
  {
    stringstream ss(line);
    string item;
    vector<string> row;

    while (getline(ss, item, ','))
    {
      row.push_back(item);
    }

    if (row.size() >= 3)
    {
      int userId = stoi(row[0]);
      int movieId = stoi(row[1]);
      float rating = stof(row[2]);

      userRatings[userId].push_back({movieId, rating});
    }
  }

  // Add users with their complete rating vectors
  for (const auto &[userId, ratings] : userRatings)
  {
    bg.addUser(userId, ratings);
  }
}

// Function to run recommendations for a specific user
vector<pair<int, float>> getRecommendationsForUser(int userId, BipartiteGraph &bg)
{
  Collaborative collaborative(bg);
  Content content(bg);
  collaborative.preComputeSimilarities();
  content.preComputeSimilarities();

  Hybrid hybrid(bg, collaborative, content);
  auto recommendations = hybrid.getRecommendations(userId);

  // Convert double scores to float
  vector<pair<int, float>> floatRecommendations;
  for (const auto &[movieId, score] : recommendations)
  {
    floatRecommendations.push_back({movieId, static_cast<float>(score)});
  }
  return floatRecommendations;
}