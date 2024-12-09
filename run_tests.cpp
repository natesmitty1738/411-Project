#include "BipartiteGraph.h"
#include "Collabrative.h"
#include "Content.h"
#include "Hybrid.h"
#include "TestUtils.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;
using namespace TestUtils;

// Helper function to print test results
void printTestResult(const string &testName, bool passed)
{
  cout << setw(60) << left << testName << ": " << (passed ? "PASSED" : "FAILED") << endl;
}

// Helper function to measure execution time
template <typename Func>
double measureExecutionTime(Func func)
{
  auto start = chrono::high_resolution_clock::now();
  func();
  auto end = chrono::high_resolution_clock::now();
  return chrono::duration_cast<chrono::milliseconds>(end - start).count();
}

// Test Suite 1: Content-Based Filtering Core Functionality
bool test_ContentBasedFiltering_SimilarGenresGetHigherScores()
{
  BipartiteGraph bg;

  // Add movies with clear genre patterns
  bg.addItem(1, {"Action", "Adventure"}, 120, 8.0, 2020);
  bg.addItem(2, {"Action", "Adventure"}, 115, 7.5, 2020);
  bg.addItem(3, {"Drama", "Romance"}, 110, 7.0, 2020);

  Content content(bg);
  content.preComputeSimilarities();

  double similarGenreScore = content.calculateSimilarity(1, 2);   // Should be high
  double differentGenreScore = content.calculateSimilarity(1, 3); // Should be low

  return similarGenreScore > differentGenreScore && similarGenreScore > 0.7;
}

bool test_ContentBasedFiltering_HandlesEmptyGenres()
{
  BipartiteGraph bg;

  bg.addItem(1, {}, 120, 8.0, 2020);
  bg.addItem(2, {"Action"}, 115, 7.5, 2020);

  Content content(bg);
  content.preComputeSimilarities();

  double similarity = content.calculateSimilarity(1, 2);
  return similarity >= 0.0 && similarity <= 1.0; // Should return valid similarity
}

// Test Suite 2: Collaborative Filtering Core Functionality
bool test_CollaborativeFiltering_SimilarUsersGetSimilarRecommendations()
{
  BipartiteGraph bg;

  // Add movies with clear patterns
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addItem(2, {"Action"}, 115, 7.5, 2020);
  bg.addItem(3, {"Drama"}, 110, 7.0, 2020);
  bg.addItem(4, {"Drama"}, 105, 6.5, 2020);

  // Add similar users with clear preferences
  bg.addUser(1, {{1, 5.0}, {2, 4.8}}); // Action fan
  bg.addUser(2, {{1, 4.9}, {2, 4.7}}); // Also action fan
  bg.addUser(3, {{3, 4.9}, {4, 4.7}}); // Drama fan

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  collab.preComputeSimilarities();

  auto rec1 = collab.getRecommendations(1);
  auto rec2 = collab.getRecommendations(2);

  // Both action fans should get similar recommendations
  double overlap = calculateRecommendationOverlap(rec1, rec2);
  cout << "Recommendation overlap between similar users: " << (overlap * 100) << "%" << endl;
  return overlap > 0.8; // High overlap expected
}

bool test_CollaborativeFiltering_HandlesNewUserWithNoRatings()
{
  BipartiteGraph bg;

  // Add movies and some users with ratings
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addItem(2, {"Action"}, 115, 7.5, 2020);
  bg.addUser(1, {{1, 5.0}, {2, 4.8}});

  // Add new user with no ratings
  bg.addUser(2, {});

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  collab.preComputeSimilarities();

  auto recs = collab.getRecommendations(2);
  return !recs.empty(); // Should still provide recommendations
}

// Test Suite 3: PageRank Influence Tests
bool test_PageRank_ActiveUsersGetHigherRank()
{
  BipartiteGraph bg;

  // Add movies
  for (int i = 1; i <= 5; i++)
  {
    bg.addItem(i, {"Action"}, 120, 8.0, 2020);
  }

  // Add active user with many ratings
  bg.addUser(1, {{1, 5.0}, {2, 4.8}, {3, 4.5}, {4, 4.2}, {5, 4.0}});

  // Add less active user
  bg.addUser(2, {{1, 4.5}});

  PageRank pageRank(bg);

  return pageRank.getPageRank(1) > pageRank.getPageRank(2);
}

bool test_PageRank_HandlesIsolatedUsers()
{
  BipartiteGraph bg;

  // Add movies
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addItem(2, {"Drama"}, 115, 7.5, 2020);

  // Add two users who rated different movies (no overlap)
  bg.addUser(1, {{1, 5.0}});
  bg.addUser(2, {{2, 4.0}});

  PageRank pageRank(bg);

  double rank1 = pageRank.getPageRank(1);
  double rank2 = pageRank.getPageRank(2);

  return rank1 > 0 && rank2 > 0; // Should assign non-zero ranks
}

// Test Suite 4: Hybrid Recommendation Integration Tests
bool test_Hybrid_CombinesAllComponents()
{
  BipartiteGraph bg;

  // Add movies with clear patterns
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addItem(2, {"Action"}, 115, 7.5, 2020);
  bg.addItem(3, {"Drama"}, 110, 7.0, 2020);

  // Add users with clear preferences
  bg.addUser(1, {{1, 5.0}, {2, 4.8}}); // Action fan
  bg.addUser(2, {{3, 4.5}});           // Drama fan

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  Content content(bg);
  collab.preComputeSimilarities();
  content.preComputeSimilarities();

  Hybrid hybrid(bg, collab, content);

  auto recs1 = hybrid.getRecommendations(1);
  auto recs2 = hybrid.getRecommendations(2);

  // Recommendations should differ based on preferences
  return calculateRecommendationOverlap(recs1, recs2) < 0.5;
}

bool test_Hybrid_HandlesEdgeCases()
{
  BipartiteGraph bg;

  // Add single movie and user
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addUser(1, {{1, 5.0}});

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  Content content(bg);
  collab.preComputeSimilarities();
  content.preComputeSimilarities();

  Hybrid hybrid(bg, collab, content);

  auto recs = hybrid.getRecommendations(1);
  return recs.empty(); // Should handle case with no unwatched movies
}

// Test PageRank influence on new users
bool test_CollaborativeFiltering_UsesPageRankForNewUsers()
{
  BipartiteGraph bg;

  // Add movies
  bg.addItem(1, {"Action"}, 120, 8.0, 2020);
  bg.addItem(2, {"Action"}, 115, 7.5, 2020);
  bg.addItem(3, {"Drama"}, 110, 7.0, 2020);

  // Add influential users with high ratings for specific movies
  for (int i = 1; i <= 5; i++)
  {                                      // Add MIN_INFLUENTIAL_USERS users
    bg.addUser(i, {{1, 5.0}, {2, 4.8}}); // These users prefer action movies
  }

  // Add some regular users with different preferences
  for (int i = 6; i <= 10; i++)
  {
    bg.addUser(i, {{3, 4.5}}); // These users prefer drama
  }

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  collab.preComputeSimilarities();

  // Add a new user
  bg.addUser(100, {});

  auto recs = collab.getRecommendations(100);

  // The new user should get recommendations influenced by high PageRank users
  // who prefer action movies
  bool foundAction = false;
  for (const auto &[movieId, score] : recs)
  {
    if (movieId == 1 || movieId == 2)
    {
      foundAction = true;
      break;
    }
  }

  return foundAction;
}

// Scaleability Test Suite
bool test_Scale_SmallStartup()
{
  BipartiteGraph bg;
  const int NUM_MOVIES = 50; // Small movie catalog
  const int NUM_USERS = 100; // Small user base
  const int AVG_RATINGS = 5; // Users rate few movies initially

  cout << "\nTesting startup scenario: " << NUM_USERS << " early users, "
       << NUM_MOVIES << " movies" << endl;

  // Add movies (mix of different genres)
  random_device rd;
  mt19937 rng(rd());
  for (int i = 1; i <= NUM_MOVIES; i++)
  {
    bg.addItem(i, generateRandomGenres(2, rng),
               90 + (i % 60),         // Length: 90-150 minutes
               6.0 + (i % 40) / 10.0, // Rating: 6.0-10.0
               i % 4);                // Mix of age ratings
  }

  // Add users with sparse ratings
  for (int i = 1; i <= NUM_USERS; i++)
  {
    int numRatings = max(1, static_cast<int>(normal_distribution<>(AVG_RATINGS, 2)(rng)));
    bg.addUser(i, generateRandomRatings(NUM_MOVIES, numRatings, rng));
  }

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  Content content(bg);

  double setupTime = measureExecutionTime([&]()
                                          {
        collab.preComputeSimilarities();
        content.preComputeSimilarities(); });
  cout << "Initial setup time: " << setupTime << "ms" << endl;

  // Test cold-start recommendations
  auto recs = collab.getRecommendations(NUM_USERS + 1); // New user
  cout << "Cold-start recommendations: " << recs.size() << " items" << endl;

  return setupTime < 5000 && !recs.empty(); // Setup under 5 seconds
}

bool test_Scale_EstablishedService()
{
  BipartiteGraph bg;
  const int NUM_MOVIES = 200; // Medium movie catalog
  const int NUM_USERS = 500;  // Medium user base
  const int AVG_RATINGS = 20; // More ratings per user

  cout << "\nTesting established service: " << NUM_USERS << " active users, "
       << NUM_MOVIES << " movies" << endl;

  // Add movies with realistic distribution
  random_device rd;
  mt19937 rng(rd());
  vector<string> genres = {"Action", "Drama", "Comedy", "Sci-Fi", "Horror",
                           "Romance", "Thriller", "Family", "Documentary"};

  for (int i = 1; i <= NUM_MOVIES; i++)
  {
    // Movies tend to have 2-3 genres
    int numGenres = normal_distribution<>(2.5, 0.5)(rng);
    numGenres = max(1, min(4, numGenres));

    bg.addItem(i, generateRandomGenres(numGenres, rng),
               90 + (i % 90),         // Length: 90-180 minutes
               6.5 + (i % 35) / 10.0, // Rating: 6.5-10.0
               i % 4);                // Mix of age ratings
  }

  // Add users with realistic rating patterns
  for (int i = 1; i <= NUM_USERS; i++)
  {
    int numRatings = max(5, static_cast<int>(normal_distribution<>(AVG_RATINGS, 8)(rng)));
    bg.addUser(i, generateRandomRatings(NUM_MOVIES, numRatings, rng));
  }

  PageRank pageRank(bg);
  Collaborative collab(bg, pageRank);
  Content content(bg);

  double setupTime = measureExecutionTime([&]()
                                          {
        collab.preComputeSimilarities();
        content.preComputeSimilarities(); });
  cout << "Setup time: " << setupTime << "ms" << endl;

  // Test recommendation quality
  vector<int> testUsers = {1, NUM_USERS / 2, NUM_USERS};
  double totalTime = 0;
  int totalRecs = 0;

  for (int userId : testUsers)
  {
    double recTime = measureExecutionTime([&]()
                                          {
            auto recs = collab.getRecommendations(userId);
            totalRecs += recs.size(); });
    totalTime += recTime;
  }

  double avgTime = totalTime / testUsers.size();
  double avgRecs = static_cast<double>(totalRecs) / testUsers.size();

  cout << "Average recommendation time: " << avgTime << "ms" << endl;
  cout << "Average recommendations per user: " << avgRecs << endl;

  return setupTime < 10000 && avgTime < 500 && avgRecs >= 5; // Setup under 10s, recs under 500ms
}

bool test_Scale_ActiveCommunity()
{
  try
  {
    BipartiteGraph bg;
    const int NUM_MOVIES = 150;       // Medium movie catalog
    const int NUM_CORE_USERS = 50;    // Dedicated users
    const int NUM_CASUAL_USERS = 250; // Casual users
    const int CORE_RATINGS = 100;     // Core users rate many movies
    const int CASUAL_RATINGS = 5;     // Casual users rate fewer movies

    cout << "\nTesting active community: " << NUM_CORE_USERS << " core users, "
         << NUM_CASUAL_USERS << " casual users" << endl;
    cout << "Core users rate ~" << CORE_RATINGS << " movies, casual users rate ~"
         << CASUAL_RATINGS << " movies" << endl;

    random_device rd;
    mt19937 rng(rd());

    // Add movies with varying popularity
    cout << "Adding " << NUM_MOVIES << " movies..." << endl;
    for (int i = 1; i <= NUM_MOVIES; i++)
    {
      float baseRating = 7.0;
      if (i <= NUM_MOVIES / 3)
        baseRating += 2.0; // Top third are highly rated
      else if (i <= 2 * NUM_MOVIES / 3)
        baseRating += 1.0; // Middle third are moderately rated

      vector<string> genres = generateRandomGenres(2, rng);
      cout << "Adding movie " << i << " with base rating " << baseRating
           << " and genres: ";
      for (const auto &genre : genres)
        cout << genre << " ";
      cout << endl;

      bg.addItem(i, genres,
                 90 + (i % 90),
                 baseRating + (i % 10) / 10.0,
                 i % 4);
    }

    // Add core users (rate many movies, focus on popular ones)
    cout << "Adding core users..." << endl;
    for (int i = 1; i <= NUM_CORE_USERS; i++)
    {
      int numRatings = normal_distribution<>(CORE_RATINGS, 10)(rng);
      numRatings = max(CORE_RATINGS / 2, min(NUM_MOVIES, numRatings));

      cout << "Core user " << i << " rating " << numRatings << " movies" << endl;

      // Core users rate more popular movies
      vector<pair<int, float>> ratings;
      vector<int> movieIds(NUM_MOVIES);
      for (int j = 0; j < NUM_MOVIES; j++)
        movieIds[j] = j + 1;

      // Bias towards rating popular movies
      sort(movieIds.begin(), movieIds.end(), [NUM_MOVIES](int a, int b)
           {
             bool a_top = (a <= NUM_MOVIES / 3);
             bool b_top = (b <= NUM_MOVIES / 3);
             if (a_top != b_top)
               return a_top; // Prioritize top third
             return a < b;   // Otherwise maintain stable order
           });

      // Rate movies with high ratings for favorites
      for (int j = 0; j < numRatings && j < NUM_MOVIES; j++)
      {
        float rating = normal_distribution<>(4.5, 0.5)(rng);
        if (movieIds[j] <= NUM_MOVIES / 3)
        { // Favorite movies get higher ratings
          rating = normal_distribution<>(4.8, 0.3)(rng);
        }
        rating = max(1.0f, min(5.0f, rating));
        ratings.push_back({movieIds[j], rating});
      }

      bg.addUser(i, ratings);
    }

    // Add casual users (rate fewer movies randomly)
    cout << "Adding casual users..." << endl;
    for (int i = NUM_CORE_USERS + 1; i <= NUM_CORE_USERS + NUM_CASUAL_USERS; i++)
    {
      int numRatings = normal_distribution<>(CASUAL_RATINGS, 2)(rng);
      numRatings = max(1, min(CASUAL_RATINGS * 2, numRatings));

      cout << "Casual user " << i << " rating " << numRatings << " movies" << endl;

      auto ratings = generateRandomRatings(NUM_MOVIES, numRatings, rng);
      bg.addUser(i, ratings);
    }

    cout << "Creating PageRank..." << endl;
    PageRank pageRank(bg);
    cout << "Creating Collaborative..." << endl;
    Collaborative collab(bg, pageRank);
    cout << "Creating Content..." << endl;
    Content content(bg);

    cout << "Computing similarities..." << endl;
    double setupTime = measureExecutionTime([&]()
                                            {
            collab.preComputeSimilarities();
            content.preComputeSimilarities(); });
    cout << "Setup time: " << setupTime << "ms" << endl;

    // Verify PageRank influence
    cout << "Calculating PageRank influence..." << endl;
    vector<pair<int, double>> userRanks;
    double totalRank = 0.0;
    double coreUserRankSum = 0.0;

    // First pass: calculate total rank
    for (int i = 1; i <= NUM_CORE_USERS + NUM_CASUAL_USERS; i++)
    {
      double rank = pageRank.getPageRank(i);
      cout << "User " << i << " rank: " << rank << endl;
      totalRank += rank;
      if (i <= NUM_CORE_USERS)
      {
        coreUserRankSum += rank;
      }
    }

    // Calculate core user influence
    double coreUserRankShare = coreUserRankSum / totalRank;
    cout << "Core users' share of total PageRank: "
         << (coreUserRankShare * 100) << "%" << endl;

    // Test recommendations reflect core users' preferences
    cout << "Testing recommendations..." << endl;
    int testUser = NUM_CORE_USERS + NUM_CASUAL_USERS + 1; // New user
    auto recs = collab.getRecommendations(testUser);

    int topMoviesRecommended = 0;
    cout << "Recommendations for new user:" << endl;
    for (const auto &[movieId, score] : recs)
    {
      cout << "Movie " << movieId << " score: " << score << endl;
      if (movieId <= NUM_MOVIES / 3)
      { // Movie is from top third
        topMoviesRecommended++;
      }
    }

    double topMovieRatio = !recs.empty() ? static_cast<double>(topMoviesRecommended) / recs.size() : 0.0;
    cout << "Ratio of top movies in recommendations: "
         << (topMovieRatio * 100) << "%" << endl;

    // Success criteria:
    // 1. Setup time is reasonable
    // 2. Core users have significant PageRank share (at least 30%)
    // 3. Recommendations reflect core users' preferences (at least 50% top movies)
    return setupTime < 10000 &&
           coreUserRankShare >= 0.3 &&
           topMovieRatio >= 0.5;
  }
  catch (const exception &e)
  {
    cout << "Exception caught: " << e.what() << endl;
    return false;
  }
  catch (...)
  {
    cout << "Unknown exception caught" << endl;
    return false;
  }
}

int main()
{
  cout << "\nRunning Test Suite..." << endl;
  cout << string(80, '=') << endl;

  vector<pair<string, bool>> results = {
      // Core functionality tests
      {"Content-Based: Similar Genres Get Higher Scores",
       test_ContentBasedFiltering_SimilarGenresGetHigherScores()},
      {"Content-Based: Handles Empty Genres",
       test_ContentBasedFiltering_HandlesEmptyGenres()},
      {"Collaborative: Handles New Users",
       test_CollaborativeFiltering_HandlesNewUserWithNoRatings()},
      {"Collaborative: Uses PageRank for New Users",
       test_CollaborativeFiltering_UsesPageRankForNewUsers()},
      {"PageRank: Active Users Get Higher Rank",
       test_PageRank_ActiveUsersGetHigherRank()},
      {"PageRank: Handles Isolated Users",
       test_PageRank_HandlesIsolatedUsers()},
      {"Hybrid: Combines All Components",
       test_Hybrid_CombinesAllComponents()},
      {"Hybrid: Handles Edge Cases",
       test_Hybrid_HandlesEdgeCases()},

      // Scale tests with realistic scenarios
      {"Scale: Startup Phase (100 users, 50 movies)",
       test_Scale_SmallStartup()},
      {"Scale: Established Service (500 users, 200 movies)",
       test_Scale_EstablishedService()},
      {"Scale: Active Community (300 users, 150 movies)",
       test_Scale_ActiveCommunity()}};

  int passed = 0;
  for (const auto &[name, result] : results)
  {
    printTestResult(name, result);
    if (result)
      passed++;
  }

  cout << "\nSummary: " << passed << "/" << results.size() << " tests passed" << endl;

  return passed == results.size() ? 0 : 1;
}