#include "BipartiteGraph.h"

using namespace std;

void BipartiteGraph::addUser(int id, const std::vector<std::pair<int, float>> &ratings)
{
  // Filter out ratings for non-existent movies
  std::vector<std::pair<int, float>> validRatings;
  validRatings.reserve(ratings.size());

  for (const auto &[movieId, rating] : ratings)
  {
    if (items.find(movieId) != items.end())
    {
      validRatings.push_back({movieId, rating});
    }
  }

  // Add user_to_items edges (only for valid movies)
  user_to_items[id] = validRatings;

  // Update item_to_users for each valid movie this user rated
  for (const auto &[movieId, rating] : validRatings)
  {
    item_to_users[movieId].push_back({id, rating});
  }
}

void BipartiteGraph::addItem(int id, vector<string> genres, int length, float imdb, int rating)
{
  Item item;
  item.id = id;
  item.genres = genres;
  item.length = length;
  item.imdb = imdb;
  item.rating = rating;
  items[id] = item; // Store the item
}

std::vector<BipartiteGraph::User> BipartiteGraph::getAllUsers() const
{
  std::vector<User> allUsers;
  for (const auto &[userId, items] : user_to_items)
  {
    User user;
    user.id = userId;

    // Get watched movies and ratings for this user
    for (const auto &[movieId, weight] : items)
    {
      // Only include ratings for existing movies
      if (this->items.find(movieId) != this->items.end())
      {
        // Assuming weight represents rating
        user.rating.push_back({movieId, weight});
        user.movie_watched.push_back({movieId, 1});
      }
    }

    allUsers.push_back(user);
  }
  return allUsers;
}