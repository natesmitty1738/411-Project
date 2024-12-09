#ifndef BIPARTITEGRAPH_H
#define BIPARTITEGRAPH_H

#include <unordered_map>
#include <vector>
#include <string>

class BipartiteGraph
{
public:
  struct User
  {
    int id;
    // movie_id, times watched
    std::vector<std::pair<int, int>> movie_watched; // movies that have already been watched should be removed from the recommendations
    // movie_id, rating
    std::vector<std::pair<int, float>> rating; // should remove one of these/combine them
    // page rank score
    float pr;
  };

  struct Item
  {
    int id;
    // genres
    std::vector<std::string> genres;
    int length;
    float imdb;
    // G, PG, PG-13, R = [0,1,2,3]
    int rating;
  };

private:
  // User -> [(Item, Weight)]
  std::unordered_map<int, std::vector<std::pair<int, float>>> user_to_items;
  // Item -> [(User, Weight)]
  std::unordered_map<int, std::vector<std::pair<int, float>>> item_to_users;
  // Item storage
  std::unordered_map<int, Item> items;

public:
  void addItem(int id, std::vector<std::string> genres, int length, float imdb, int rating);
  void addUser(int id, const std::vector<std::pair<int, float>> &ratings);
  std::vector<User> getAllUsers() const;

  const std::unordered_map<int, std::vector<std::pair<int, float>>> &getUserItems() const
  {
    return user_to_items;
  }

  const std::unordered_map<int, std::vector<std::pair<int, float>>> &getItemUsers() const
  {
    return item_to_users;
  }

  const std::unordered_map<int, Item> &getItems() const
  {
    return items;
  }
};

#endif