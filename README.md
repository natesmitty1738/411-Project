# Movie Recommendation System: A Hybrid Approach with PageRank at its Core

```bash
# Clone
git clone git@github.com:natesmitty1738/411-Project.git
cd 411-Project

# Build & test
make && ./run_tests
```

## Table of Contents

1. [Test Cases](#test-cases)
2. [Introduction](#introduction)
3. [Implementation](#implementation)
   - [1. BipartiteGraph: The Foundation](#1-bipartitegraph-the-foundation)
   - [2. Content-Based Filtering](#2-content-based-filtering-personalized-genre-based-recommendations)
   - [3. Collaborative Filtering](#3-collaborative-filtering-user-based-with-pagerank-integration)
   - [4. Utils: Core Similarity Metrics](#4-utils-core-similarity-metrics)
   - [5. Hybrid: Final Recommendation Generation](#5-hybrid-final-recommendation-generation)
4. [Run Time Analysis](#run-time-analysis)
   - [PageRank Component Analysis](#pagerank-component-analysis)
   - [Content-Based Component Analysis](#content-based-component-analysis)
   - [Collaborative Component Analysis](#collaborative-component-analysis)
   - [Hybrid Component Analysis](#hybrid-component-analysis)
5. [Performance Visualizations](#performance-visualizations)
6. [Feedback](#feedback)
7. [Citations](#citations)

## Test Cases

1. **Content-Based Tests**
   - `test_ContentBasedFiltering_SimilarGenresGetHigherScores`: Movies with matching genres have higher similarity scores
   - `test_ContentBasedFiltering_HandlesEmptyGenres`: System properly handles movies with no genres

2. **Collaborative Tests**
   - `test_CollaborativeFiltering_SimilarUsersGetSimilarRecommendations`: Users with similar ratings get similar recommendations
   - `test_CollaborativeFiltering_HandlesNewUserWithNoRatings`: System can handle new users
   - `test_CollaborativeFiltering_UsesPageRankForNewUsers`: Recommendations for new users are influenced by high PageRank users

3. **PageRank Tests**
   - `test_PageRank_ActiveUsersGetHigherRank`: Users who rate more movies get higher PageRank scores
   - `test_PageRank_HandlesIsolatedUsers`: Users with no overlapping movies

4. **Hybrid Tests**
   - `test_Hybrid_CombinesAllComponents`: Integration of collaborative, content-based, and PageRank scores
   - `test_Hybrid_HandlesEdgeCases`: Cold-start

5. **Scale Tests**
   - `test_Scale_SmallStartup`: 100 users, 50 movies
   - `test_Scale_EstablishedService`: 500 users, 200 movies
   - `test_Scale_ActiveCommunity`: mix of core (50) and casual (250) users

### Modifying Test Sizes

To modify test sizes, edit constants in `run_tests.cpp`:

```cpp
// In test_Scale_SmallStartup():
const int NUM_MOVIES = 50;        // Change number of movies
const int NUM_USERS = 100;        // Change number of users
const int AVG_RATINGS = 5;        // Change average ratings per user

// In test_Scale_EstablishedService():
const int NUM_MOVIES = 200;       // Medium catalog size
const int NUM_USERS = 500;        // Medium user base
const int AVG_RATINGS = 20;       // More ratings per user

// In test_Scale_ActiveCommunity():
const int NUM_CORE_USERS = 50;    // Number of power users
const int NUM_CASUAL_USERS = 250; // Number of casual users
const int CORE_RATINGS = 100;     // Ratings by power users
const int CASUAL_RATINGS = 5;     // Ratings by casual users
```

## Introduction

I think the value in personalized recommendations is a symptom of the brains propensity towards wanting to feel understood. Dating back to the Renaissance era with Gerolamo Cardano (1501-1576), an intellectual, who saw the value in probabilistic thinking. His favorite activity was participating in "games of chance" (he was a degenerate gambler), consequently he was passionate about honing his prediction skills. He wrote: "'Liber de Ludo Aleae' '(The Book on Games of Chance)'," which layed the groundwork for recommendation/probability science. Later in life, his magnum opes: "The Astrology Handbook," featured 66 birth horoscopes, representing one of the first documented attempts to combine data (star positions) with personal characteristics to make predictions - nowdays it's called Astrology. Of course, Astrology is complete psuedoscience. Yet, its continued existance and popularity perfectly encapsulates the demand for recommendations. A demand that is probably deeply rooted in fascinating psychology that I know nothing about.

Current Relevance:
The modern era of recommendation systems began in the 1990s at the MIT Media Lab. Pattie Maes and her team developed one of the first practical collaborative filtering systems. Their project HOMR (later Ringo) evolved into Firefly, the first commercial music recommendation system. It had 3 million users by 1996, rivaling AOL's 4.5 million. It was big. This innovation subsequently sparked a revolution in e-commerce. Well stated in an "MIT Press" article, "Firefly's success got Barnes & Noble to cut a books-based recommender deal. This made a young Amazon nervous so it cut a comparable deal with academic start-up Net Perceptions, which had been founded a year after Firefly. The rest, as they say, is history" [^1].

Let's get started:
My project implements a movie recommendation service using C++. It combines hybrid filtering with a modified PageRank algorithm to create, effective, efficient, personalized recommendations. Considering my use of multiple algorithms, I will focus on PageRank as my chosen algorithm since it's the most integral.

## Implementation

My hybrid approach combines three metrics:

- **Collaborative Filtering**: User behavior patterns and similarities
- **Content-Based Filtering**:  Movie attributes: genres, ratings, and release years
- **PageRank**: Weighs user influence based on their activity and rating patterns

This aims to negate common challenges in less advanced recommendation systems:

1. The cold-start problem (lacking data for new services)
2. Scalability
3. Balance between popularity and personalization
4. Proper influence of user engagement on recommendations

My design works as such:

1. **Data Input**
   - Movies: (id, genres[], length, rating, year)
   - Users: (id, [(movieId, rating)])
   → Stored in Bipartite Graph structure

2. **Preprocessing**
   - Movie Similarities: Jaccard Algorithm(genres) + normalized(rating, year, length)
   - User Similarities: Cosine Similarity Algorithm
   - User Influence: PageRank Algorithm
   → Preprocessing results cached for O(1) lookup (important for user experience)

3. **User Profile Analysis**
   - Genre Preferences: WeightedAverage(ratings per genre)
   - Rating Patterns: sqrt(frequency) × average_rating
   - Influence Score: PageRank based on shared movies
   → Creates user preference list

4. **Recommendation Generation** [O(M log M)]
   - Content Score: 0.8 × genre_match + 0.2 × movie_quality [O(1) with cache]
   - Collaborative Score: WeightedAverage(similar_users × pagerank) [O(1) with cache]
   - Hybrid Score: (0.6 × (1 + pagerank)) × collab + 0.4 × content [O(1)]
   → Returns top-N ranked movies after O(M log M) sort
   where M = number of unwatched movies

5. **Score Normalization**
   - All scores normalized to [0,1] range
     - this is incredibly important for large data sets
     - I ran into countless issues without it and learned:
       - Rec. metrics need to be uniform b/c of storage, out of bounds issues, and future calculations
   - Calculate weights:
        collab_weight = 0.6 × (1 + pagerank)
        content_weight = 0.4
   - Get user's PageRank (cached O(1))
     - PR(u) = (1-0.85)/N + 0.85 × Σ(PR(v) × shared_movies / total_movies_v)
   - Weights adjusted by user PageRank:

     | User Type | PageRank | Raw Weights (C/T) | Normalized (C/T) | Example Score                   |
     | --------- | -------- | ----------------- | ---------------- | ------------------------------- |
     | Power     | 0.15     | 0.69 / 0.4 = 1.09 | 63.3% / 36.7%    | 0.633×0.85 + 0.367×0.70 = 0.793 |
     | Average   | 0.05     | 0.63 / 0.4 = 1.03 | 61.2% / 38.8%    | 0.612×0.85 + 0.388×0.70 = 0.792 |
     | Casual    | 0.02     | 0.61 / 0.4 = 1.01 | 60.5% / 39.5%    | 0.605×0.85 + 0.395×0.70 = 0.790 |
     | New       | 0.001    | 0.60 / 0.4 = 1.00 | 60.0% / 40.0%    | 0.600×0.85 + 0.400×0.70 = 0.790 |

     *Where C = Collaborative weight, T = Content weight*  
     *Example uses collaborative score = 0.85, content score = 0.70*

   - Normalize weights:
        total = collab_weight + content_weight
        collab_weight /= total
        content_weight /= total
   - Final scores sorted for ranking
   → Produces final recommendations ordered by hybrid score

## Intuition

### 1. BipartiteGraph: The Foundation

The BipartiteGraph class serves as the foundational data structure for the entire recommendation system. It models the relationships between users and movies as a bipartite graph, where connections only exist between users and movies (never user-to-user or movie-to-movie).

#### Class Structure

```pseudocode
class BipartiteGraph
    // Core data structures
    user_to_items: HashMap<int, List<Pair<int, float>>>  // User ratings
    item_to_users: HashMap<int, List<Pair<int, float>>>  // Movie ratings
    items: HashMap<int, Item>                            // Movie attributes

    // Item structure
    struct Item
        genres: List<string>
        length: int
        imdb: float
        year: int

    // Core functions
    function addUser(userId, ratings)
    function addItem(movieId, genres, length, rating, year)
    function getUserItems() const
    function getItemUsers() const
    function getItems() const
```

#### Implementation Details

1. **Data Storage**:

```cpp
// User rating storage
user_to_items[1] = {
    {101, 4.5},  // Movie 101, rating 4.5
    {102, 3.0},  // Movie 102, rating 3.0
    ...
}

// Movie metadata storage
items[101] = {
    genres: ["Action", "Adventure"],
    length: 120,
    imdb: 8.5,
    year: 2020
}
```

1. **Rating Addition**:

```cpp
void addUser(userId, ratings) {
    for each (movieId, rating) in ratings {
        // Add to user's ratings
        user_to_items[userId].push_back({movieId, rating})
        
        // Add to movie's ratings (bidirectional access)
        item_to_users[movieId].push_back({userId, rating})
    }
}
```

1. **Movie Addition**:

```cpp
void addItem(movieId, genres, length, rating, year) {
    items[movieId] = Item{
        genres: genres,
        length: length,
        imdb: rating,
        year: year
    }
}
```

#### Data Flow Example

Let's follow how data flows through the BipartiteGraph when loading a movie and user ratings:

1. **Initial Movie Load**:

```cpp
// From run_tests.cpp test data
bg.addItem(1, {"Action", "Adventure"}, 120, 8.0, 2020);
bg.addItem(2, {"Action", "Adventure"}, 115, 7.5, 2020);
bg.addItem(3, {"Drama", "Romance"}, 110, 7.0, 2020);

// Internal state after loading:
items = {
    1: {genres: ["Action", "Adventure"], length: 120, imdb: 8.0, year: 2020},
    2: {genres: ["Action", "Adventure"], length: 115, imdb: 7.5, year: 2020},
    3: {genres: ["Drama", "Romance"], length: 110, imdb: 7.0, year: 2020}
}
```

1. **User Rating Addition**:

```cpp
// From test data: Adding an action fan
bg.addUser(1, {{1, 5.0}, {2, 4.8}});

// Internal state after addition:
user_to_items = {
    1: [{1, 5.0}, {2, 4.8}]  // User 1's ratings
}

item_to_users = {
    1: [{1, 5.0}],  // Movie 1's ratings
    2: [{1, 4.8}]   // Movie 2's ratings
}
```

1. **Data Access Patterns**:

```cpp
// For content-based filtering:
items[movieId].genres  // O(1) access to movie attributes

// For collaborative filtering:
user_to_items[userId]  // O(1) access to user's ratings

// For PageRank calculation:
item_to_users[movieId] // O(1) access to movie's ratings
```

This graph structure enables:

- Efficient movie similarity calculations (content-based)
- Quick user rating pattern analysis (collaborative)
- Fast shared movie identification (PageRank)
- O(1) average-case lookups for all operations

### 2. Content-Based Filtering: Personalized Genre-Based Recommendations

The Content class implements content-based filtering approach that focuses on movie attributes, particularly genres. The implementation is found in `content.h` and `content.cpp`.

#### Class Structure

```pseudocode
class Content
    // Core data structures
    graph: const BipartiteGraph&         // Reference to data graph
    similarityCache: HashMap<uint64, float>     // Cached similarity scores
    cacheAccessCount: HashMap<uint64, int>      // Cache usage tracking
    MAX_CACHE_SIZE = 10000                      // Prevent memory overflow

    // Core functions
    function calculateSimilarity(item1Id, item2Id) const
    function preComputeSimilarities(numThreads = 4)
    function getRecommendations(userId, n = 10) const
    function getSimilarItems(itemId, n = 5) const
```

#### Implementation Details

1. **Movie Similarity Calculation**:

```cpp
float calculateSimilarity(item1Id, item2Id) {
    if (item1Id == item2Id) return 1.0f;
    
    const auto& item1 = items[item1Id];
    const auto& item2 = items[item2Id];
    
    // Genre similarity (Jaccard coefficient)
    float genreSimilarity = Utils::jaccardSimilarity(
        item1.genres, item2.genres
    );
    
    // Normalize other attributes to [0,1]
    float ratingDiff = abs(item1.imdb - item2.imdb) / 10.0f;
    float yearDiff = abs(item1.year - item2.year) / 4.0f;
    float lengthDiff = abs(item1.length - item2.length) / 180.0f;
    
    // Weighted combination
    return 0.6f * genreSimilarity +
           0.2f * (1.0f - ratingDiff) +
           0.1f * (1.0f - yearDiff) +
           0.1f * (1.0f - lengthDiff);
}
```

1. **User Preference Learning**:

```cpp
// From getRecommendations() implementation
void analyzeUserPreferences(userId, genrePreferences) {
    // Count genre occurrences and ratings
    for (const auto& [movieId, rating] : userRatings) {
        for (const auto& genre : items[movieId].genres) {
            genreStats[genre].first += rating;    // Sum of ratings
            genreStats[genre].second++;           // Count of ratings
        }
    }
    
    // Calculate weighted preferences
    for (const auto& [genre, stats] : genreStats) {
        if (stats.second > 0) {
            float avgRating = stats.first / stats.second;
            // Weight by sqrt(count) to balance frequency
            float preference = avgRating * sqrt(stats.second);
            genrePreferences[genre] = preference;
        }
    }
}
```

1. **Recommendation Generation**:

```cpp
vector<pair<int, float>> getRecommendations(userId, n) {
    // Get user's genre preferences
    unordered_map<string, float> genrePreferences;
    analyzeUserPreferences(userId, genrePreferences);
    
    // Score unwatched movies
    vector<pair<int, float>> recommendations;
    for (const auto& [movieId, movie] : items) {
        if (watchedMovies.count(movieId) > 0) continue;
        
        // Calculate genre score
        float genreScore = 0.0f;
        float totalWeight = 0.0f;
        for (const auto& genre : movie.genres) {
            if (genrePreferences.count(genre) > 0) {
                genreScore += genrePreferences[genre];
                totalWeight += 1.0f;
            }
        }
        
        if (totalWeight > 0) genreScore /= totalWeight;
        
        // Combine with movie quality
        float score = 0.8f * genreScore + 
                     0.2f * (movie.imdb / 10.0f);
        recommendations.push_back({movieId, score});
    }
    
    // Sort and return top-N
    sort(recommendations.begin(), recommendations.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
    if (recommendations.size() > n) 
        recommendations.resize(n);
    return recommendations;
}
```

#### Data Flow Example

1. **Initial State**:

```cpp
// From test_ContentBasedFiltering_SimilarGenresGetHigherScores
BipartiteGraph bg;
bg.addItem(1, {"Action", "Adventure"}, 120, 8.0, 2020);
bg.addItem(2, {"Action", "Adventure"}, 115, 7.5, 2020);
bg.addItem(3, {"Drama", "Romance"}, 110, 7.0, 2020);

Content content(bg);
content.preComputeSimilarities();
```

1. **Similarity Calculation**:

```cpp
// Calculate similarity between movies 1 and 2
float similarity = content.calculateSimilarity(1, 2);

// Internal calculation:
genreSimilarity = 2/2 = 1.0           // Same genres
ratingDiff = |8.0 - 7.5| / 10 = 0.05  // Similar ratings
yearDiff = |2020 - 2020| / 4 = 0.0    // Same year
lengthDiff = |120 - 115| / 180 = 0.028 // Similar length

finalSimilarity = 0.6 * 1.0 + 
                 0.2 * (1 - 0.05) + 
                 0.1 * (1 - 0.0) + 
                 0.1 * (1 - 0.028)
                = 0.6 + 0.19 + 0.1 + 0.0972
                ≈ 0.987  // High similarity
```

1. **Recommendation Generation**:

```cpp
// Add a user who likes action movies
bg.addUser(1, {{1, 5.0}, {2, 4.8}});

auto recs = content.getRecommendations(1);

// Internal calculation:
genrePreferences = {
    "Action": 4.9 * sqrt(2) ≈ 6.93,     // High preference
    "Adventure": 4.9 * sqrt(2) ≈ 6.93    // High preference
}

// Movie 3 score calculation:
genreScore = 0.0  // No matching genres
qualityScore = 7.0 / 10.0 = 0.7
finalScore = 0.8 * 0.0 + 0.2 * 0.7 = 0.14  // Low score
```

This example demonstrates how the content-based filtering:
- Uses math to identify similar movies based on attributes
- Learns user preferences from rating patterns
- Weights recommendations by both preferences and quality
- Effectively differentiates between genres

### 3. Collaborative Filtering: User-Based with PageRank Integration

The Collaborative class implements a sophisticated user-based collaborative filtering system that incorporates PageRank scores to weight user influences. The implementation is found in `collabrative.h` and `collabrative.cpp`.

#### Class Structure

```pseudocode
class Collaborative
    // Core data structures
    graph: const BipartiteGraph&         // Reference to data graph
    pageRank: const PageRank&           // User influence scores
    similarityCache: HashMap<uint64, float>     // User similarity scores
    cacheAccessCount: HashMap<uint64, int>      // Cache usage tracking
    
    // Configuration constants
    MAX_CACHE_SIZE = 10000              // Cache size limit
    MIN_INFLUENTIAL_USERS = 5           // For PageRank-based recs
    MIN_PAGERANK_SCORE = 0.01          // Minimum influence threshold

    // Core functions
    function calculateSimilarity(user1Id, user2Id) const
    function preComputeSimilarities(numThreads)
    function getRecommendations(userId, n = 5) const
    function getInfluentialRecommendations(userMovies, n) const
```

#### Implementation Details

1. **User Similarity Calculation**:

```cpp
float calculateSimilarity(user1Id, user2Id) {
    if (user1Id == user2Id) return 1.0f;
    
    const auto& ratings1 = users[user1Id];
    const auto& ratings2 = users[user2Id];
    
    // Create O(1) lookup map
    unordered_map<int, float> user1Ratings;
    for (const auto& [movieId, rating] : ratings1) {
        user1Ratings[movieId] = rating;
        norm1 += rating * rating;
    }
    
    // Calculate dot product and norms
    double dotProduct = 0.0, norm2 = 0.0;
    int commonMovies = 0;
    
    for (const auto& [movieId, rating2] : ratings2) {
        auto it = user1Ratings.find(movieId);
        if (it != user1Ratings.end()) {
            dotProduct += it->second * rating2;
            commonMovies++;
        }
        norm2 += rating2 * rating2;
    }
    
    // Require at least one common movie
    if (commonMovies < 1) return 0.0f;
    
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}
```

1. **Recommendation Generation for Existing Users**:

```cpp
vector<pair<int, float>> getRecommendations(userId, n) {
    // Find similar users
    vector<pair<int, float>> similarUsers;
    for (const auto& [otherId, _] : users) {
        if (otherId == userId) continue;
        float similarity = getCachedSimilarity(userId, otherId);
        if (similarity > 0) {
            similarUsers.push_back({otherId, similarity});
        }
    }
    
    // Sort and take top-K similar users
    sort(similarUsers.begin(), similarUsers.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
    if (similarUsers.size() > 10) 
        similarUsers.resize(10);
    
    // Get weighted recommendations
    unordered_map<int, pair<float, float>> scores;
    for (const auto& [otherId, similarity] : similarUsers) {
        float weight = similarity * pageRank.getPageRank(otherId);
        for (const auto& [movieId, rating] : users[otherId]) {
            if (watchedMovies.count(movieId) > 0) continue;
            scores[movieId].first += rating * weight;
            scores[movieId].second += weight;
        }
    }
    
    // Calculate final scores
    vector<pair<int, float>> recommendations;
    for (const auto& [movieId, weights] : scores) {
        if (weights.second > 0) {
            float score = weights.first / weights.second;
            // Blend with movie quality
            score = 0.8f * score + 
                   0.2f * items[movieId].imdb;
            recommendations.push_back({movieId, score});
        }
    }
    
    // Sort and return top-N
    sort(recommendations.begin(), recommendations.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
    if (recommendations.size() > n)
        recommendations.resize(n);
    return recommendations;
}
```

1. **New User Handling**:

```cpp
vector<pair<int, float>> getInfluentialRecommendations(
    const unordered_set<int>& userMovies, size_t n) {
    
    // Get users sorted by PageRank
    vector<pair<int, double>> usersByRank;
    for (const auto& [userId, _] : users) {
        double rank = pageRank.getPageRank(userId);
        if (rank >= MIN_PAGERANK_SCORE) {
            usersByRank.push_back({userId, rank});
        }
    }
    
    sort(usersByRank.begin(), usersByRank.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
    
    // Get recommendations from influential users
    unordered_map<int, pair<float, float>> weightedRecs;
    for (const auto& [userId, rank] : usersByRank) {
        float weight = static_cast<float>(rank);
        for (const auto& [movieId, rating] : users[userId]) {
            if (userMovies.count(movieId) > 0) continue;
            weightedRecs[movieId].first += rating * weight;
            weightedRecs[movieId].second += weight;
        }
    }
    
    // Calculate final scores
    vector<pair<int, float>> recommendations;
    for (const auto& [movieId, weights] : weightedRecs) {
        if (weights.second > 0) {
            float score = weights.first / weights.second;
            score = 0.7f * score + 
                   0.3f * items[movieId].imdb;
            recommendations.push_back({movieId, score});
        }
    }
    
    // Sort and return top-N
    sort(recommendations.begin(), recommendations.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
    if (recommendations.size() > n)
        recommendations.resize(n);
    return recommendations;
}
```

#### Data Flow Example

1. **Initial State**:

```cpp
// From test_CollaborativeFiltering_SimilarUsersGetSimilarRecommendations
BipartiteGraph bg;
bg.addItem(1, {"Action"}, 120, 8.0, 2020);
bg.addItem(2, {"Action"}, 115, 7.5, 2020);
bg.addItem(3, {"Drama"}, 110, 7.0, 2020);
bg.addItem(4, {"Drama"}, 105, 6.5, 2020);

// Add similar users with clear preferences
bg.addUser(1, {{1, 5.0}, {2, 4.8}});  // Action fan
bg.addUser(2, {{1, 4.9}, {2, 4.7}});  // Also action fan
bg.addUser(3, {{3, 4.9}, {4, 4.7}});  // Drama fan

PageRank pageRank(bg);
Collaborative collab(bg, pageRank);
collab.preComputeSimilarities();
```

1. **Similarity Calculation**:

```cpp
// Calculate similarity between users 1 and 2
float similarity = collab.calculateSimilarity(1, 2);

// Internal calculation:
user1Ratings = {1: 5.0, 2: 4.8}
norm1 = 5.0² + 4.8² = 48.04

dotProduct = (5.0 * 4.9) + (4.8 * 4.7) = 47.06
norm2 = 4.9² + 4.7² = 46.1

similarity = 47.06 / sqrt(48.04 * 46.1)
          ≈ 0.985  // Very high similarity
```

1. **Recommendation Generation**:

```cpp
auto recs = collab.getRecommendations(1);

// Internal steps:
// a. Similar users (with PageRank weights):
similarUsers = [
    {2, 0.985 * 0.4},  // User 2 with high similarity
    {3, 0.1 * 0.3}     // User 3 with low similarity
]

// b. Score calculation for movie 3:
weightedSum = (4.9 * 0.394) = 1.93
weightTotal = 0.394
score = 1.93 / 0.394 = 4.9

// c. Final score blending:
finalScore = 0.8 * 4.9 + 0.2 * 7.0
          = 3.92 + 1.4
          = 5.32
```

My collaborative filtering:
- Accurately identifies similar users through rating patterns
- Weights recommendations by both similarity and influence
- Handles the cold-start problem with PageRank-based recommendations
- Balances user preferences with movie quality

### Weight Adjustments by PageRank

The system dynamically adjusts recommendation weights based on user PageRank scores. Here's a detailed breakdown:

#### Base Configuration

```cpp
Initial Weights:
- Collaborative: 60% (0.6)
- Content-based: 40% (0.4)
```

#### Weight Adjustment Formula

```cpp
adjusted_collab = 0.6 * (1 + pagerank)
adjusted_content = 0.4  // Remains constant
final_collab = adjusted_collab / (adjusted_collab + adjusted_content)
final_content = adjusted_content / (adjusted_collab + adjusted_content)
```

#### Example Scenarios

1. **Power User** (PageRank = 0.15)

```cpp
Calculation:
- Collaborative: 0.6 * (1 + 0.15) = 0.69
- Content: 0.4
- Total = 1.09

Normalized Weights:
- Collaborative: 0.69/1.09 = 63.3%
- Content: 0.4/1.09 = 36.7%
```

1. **Average User** (PageRank = 0.05)

```cpp
Calculation:
- Collaborative: 0.6 * (1 + 0.05) = 0.63
- Content: 0.4
- Total = 1.03

Normalized Weights:
- Collaborative: 0.63/1.03 = 61.2%
- Content: 0.4/1.03 = 38.8%
```

1. **Casual User** (PageRank = 0.02)

```cpp
Calculation:
- Collaborative: 0.6 * (1 + 0.02) = 0.612
- Content: 0.4
- Total = 1.012

Normalized Weights:
- Collaborative: 0.612/1.012 = 60.5%
- Content: 0.4/1.012 = 39.5%
```

1. **New User** (PageRank = 0.001)

```cpp
Calculation:
- Collaborative: 0.6 * (1 + 0.001) = 0.6006
- Content: 0.4
- Total = 1.0006

Normalized Weights:
- Collaborative: 0.6006/1.0006 = 60.0%
- Content: 0.4/1.0006 = 40.0%
```

| PageRank | Collaborative | Content | Explanation  |
| -------- | ------------- | ------- | ------------ |
| 0.150    | 63.3%         | 36.7%   | Power user   |
| 0.050    | 61.2%         | 38.8%   | Average user |
| 0.020    | 60.5%         | 39.5%   | Casual user  |
| 0.001    | 60.0%         | 40.0%   | New user     |

#### Analysis

1. **Power Users** (PR > 0.1)
   - Highest collaborative weight (>63%)
   - Recommendations heavily influenced by similar users
   - Content preferences act as secondary filter
   - Rationale: Established taste patterns, reliable rating history

2. **Average Users** (0.03 < PR < 0.1)
   - Balanced but collaborative-leaning weights
   - Good mix of similar users and content matching
   - Rationale: Sufficient history for collaborative, but content still important

3. **Casual Users** (0.01 < PR < 0.03)
   - Slightly higher collaborative weight
   - Content matching plays significant role
   - Rationale: Limited but useful rating history

4. **New Users** (PR < 0.01)
   - Almost equal weights
   - Slight collaborative edge from high-PageRank users
   - Strong content influence
   - Rationale: Limited personal history, need content backup

This weighting system ensures:
- Smooth transition from content to collaborative dominance
- Automatic adaptation to user engagement
- Protection against cold-start problems
- Leverage of experienced users' preferences

## PageRank Implementation Details

Uses weighted edges based on shared movie ratings between users.

### Graph Construction

1. **Nodes**: Each user in the system represents a node in the graph
2. **Edges**: Edges are formed between users who have rated common movies
3. **Edge Weights**: Calculated as:

   ```cpp
   weight(user1 -> user2) = number_of_shared_movies / total_movies_user2
   ```

### Understanding User Types Through PageRank

```cpp
PR(u) = (1-d)/N + d × Σ(PR(v) × shared_movies / total_movies_v)

where:
d = damping factor (0.85) // 15% chance that a user randomly selects something out of the ordinary
// the creators (Larry Page and Sergey Brin) decided:
// this was a good standard deviation to counteract random "wrong choice" clicks
N = total number of users
PR(v) = PageRank score of user v
shared_movies = movies rated by both u and v
total_movies_v = total movies rated by user v
```

PageRank on a system with 1000 users (N=1000):

1. Power User (PR ≈ 0.15)

```cpp
Profile:
- Rates 100 movies
- Shares ratings with 200 other users
- Average 40 shared movies per connection

Calculation:
PR(power) = 0.15/1000 + 0.85 × Σ(PR(v) × 40/100)
          = 0.00015 + 0.85 × (200 × 0.05 × 0.4)
          = 0.00015 + 0.85 × 4
          ≈ 0.15
```

#### 2. Average User (PR ≈ 0.05)

```cpp
Profile:
- Rates 30 movies
- Shares ratings with 50 other users
- Average 10 shared movies per connection

Calculation:
PR(average) = 0.15/1000 + 0.85 × Σ(PR(v) × 10/30)
            = 0.00015 + 0.85 × (50 × 0.05 × 0.33)
            = 0.00015 + 0.85 × 0.825
            ≈ 0.05
```

#### 3. Casual User (PR ≈ 0.02)

```cpp
Profile:
- Rates 10 movies
- Shares ratings with 20 other users
- Average 3 shared movies per connection

Calculation:
PR(casual) = 0.15/1000 + 0.85 × Σ(PR(v) × 3/10)
           = 0.00015 + 0.85 × (20 × 0.05 × 0.3)
           = 0.00015 + 0.85 × 0.3
           ≈ 0.02
```

#### 4. New User (PR ≈ 0.001)

```cpp
Profile:
- Rates 2 movies
- Shares ratings with 3 other users
- Average 1 shared movie per connection

Calculation:
PR(new) = 0.15/1000 + 0.85 × Σ(PR(v) × 1/2)
        = 0.00015 + 0.85 × (3 × 0.05 × 0.5)
        = 0.00015 + 0.85 × 0.075
        ≈ 0.001
```

### Impact on Recommendation Weights

The PageRank scores directly influence the weight given to collaborative vs. content-based filtering:

```cpp
collaborative_weight = 0.6 × (1 + pagerank)
content_weight = 0.4 (constant)

Then normalize:
total = collaborative_weight + content_weight
final_collaborative = collaborative_weight / total
final_content = content_weight / total
```

This creates the weight distribution:

| User Type | PageRank | Raw Weights (C/T) | Normalized (C/T) | Explanation                                           |
| --------- | -------- | ----------------- | ---------------- | ----------------------------------------------------- |
| Power     | 0.15     | 0.69 / 0.4 = 1.09 | 63.3% / 36.7%    | High influence, heavily weighted toward collaborative |
| Average   | 0.05     | 0.63 / 0.4 = 1.03 | 61.2% / 38.8%    | Moderate influence, balanced weighting                |
| Casual    | 0.02     | 0.61 / 0.4 = 1.01 | 60.5% / 39.5%    | Limited influence, slightly more content-based        |
| New       | 0.001    | 0.60 / 0.4 = 1.00 | 60.0% / 40.0%    | Minimal influence, strongest content-based weight     |

### Implementation Components

1. **Initialization**
   - All users start with equal PageRank: `1/N`
   - Edge weights are pre-computed and cached
   - Damping factor set to 0.85 (standard value)

2. **Iteration Process**

   ```cpp
   for each iteration:
       for each user u:
           sum = 0
           for each neighbor v of u:
               shared = number of movies rated by both u and v
               total = total movies rated by v
               weight = shared/total
               sum += currentPR[v] * weight
           
           newPR[u] = (1-0.85)/N + 0.85 * sum
   ```

3. **Convergence Check**
   - After each iteration, check if scores have stabilized
   - Convergence threshold: 0.0001
   - Maximum iterations: 100 (rarely reached)

### Optimization Techniques

1. **Edge Weight Caching**

   ```cpp
   struct EdgeWeight {
       int shared_movies;
       int total_movies;
       float weight;
   };
   unordered_map<pair<int,int>, EdgeWeight> weight_cache;
   ```

2. **Sparse Matrix Representation**
   - Only store non-zero edges
   - Reduces memory usage and computation time
   - Particularly effective for sparse user-movie matrices

3. **Early Stopping**

   ```cpp
   if (max_diff < epsilon) {
       iterations_until_convergence = iter;
       break;
   }
   ```

### Score Interpretation

PageRank scores are interpreted as user influence levels:

| Score Range | User Category | Influence Level |
| ----------- | ------------- | --------------- |
| > 0.1       | Power User    | Very High       |
| 0.05 - 0.1  | Active User   | High            |
| 0.01 - 0.05 | Regular User  | Medium          |
| < 0.01      | New/Casual    | Low             |

### Impact on Recommendations

1. **Direct Impact**
   - Higher PageRank users' ratings carry more weight
   - Influences collaborative filtering component
   - Affects hybrid weight calculation
   - Most importantly, pagerank excels at identifying users who are:
     - highly selective, rather than broad
     - highly influential (often selects popular interests)
     - highly active (ratings have validity)

2. **Weight Adjustment**

   ```cpp
   collaborative_weight = base_weight * (1 + pagerank_score)
   ```

3. **Cold Start Handling**
   - New users initially rely more on high PageRank users
   - Gradually shifts as they build their own profile

### Performance Characteristics

1. **Time Complexity**
   - Per iteration: O(E) where E = number of edges
   - Total: O(k * E) where k = iterations until convergence
   - Typically converges in 15-20 iterations

2. **Space Complexity**
   - O(N) for PageRank vectors
   - O(E) for edge weight cache
   - Where N = number of users, E = number of edges

3. **Memory Usage**
   - Approximately 8 bytes per user (PageRank score)
   - 12 bytes per edge (source, target, weight)
   - Cache overhead: ~24 bytes per edge

### Example Calculation

For a small network with 3 users:

```cpp
User1: rated movies [1,2,3,4]
User2: rated movies [1,2,3]
User3: rated movies [3,4]

Edge Weights:
User1 -> User2: 3/3 = 1.0
User2 -> User1: 3/4 = 0.75
User1 -> User3: 2/2 = 1.0
User3 -> User1: 2/4 = 0.5
User2 -> User3: 1/2 = 0.5
User3 -> User2: 1/3 = 0.33

Initial PageRank = [0.33, 0.33, 0.33]

After convergence:
PageRank ≈ [0.42, 0.31, 0.27]
```

This ensures that:
1. User influence is accurately captured
2. Recommendations benefit from network effects
3. The system remains computationally efficient
4. Cold-start problems are effectively mitigated

## Pseudocode

### Core Algorithm Flow

```pseudocode
function getRecommendations(userId, numRecommendations):
    // Get user's PageRank score
    userRank = pageRank.getScore(userId)
    
    // Get recommendations from both approaches
    contentRecs = content.getRecommendations(userId, numRecommendations)
    collabRecs = collaborative.getRecommendations(userId, numRecommendations)
    
    // Calculate weights based on PageRank
    collabWeight = 0.6 * (1 + userRank)
    contentWeight = 0.4
    
    // Normalize weights
    totalWeight = collabWeight + contentWeight
    collabWeight /= totalWeight
    contentWeight /= totalWeight
    
    // Merge recommendations
    finalScores = new HashMap<movieId, score>
    
    for each (movieId, score) in contentRecs:
        finalScores[movieId] = contentWeight * score
        
    for each (movieId, score) in collabRecs:
        if movieId in finalScores:
            finalScores[movieId] += collabWeight * score
        else:
            finalScores[movieId] = collabWeight * score
    
    // Sort and return top N
    return sortByValueDescending(finalScores)
        .take(numRecommendations)

function calculatePageRank(graph, damping = 0.85, epsilon = 0.0001):
    numUsers = graph.getUserCount()
    ranks = new Array[numUsers].fill(1/numUsers)
    newRanks = new Array[numUsers]
    
    while true:
        // Calculate new ranks
        for each user in graph:
            sum = 0
            for each neighbor in user.getNeighbors():
                neighborRank = ranks[neighbor.id]
                edgeWeight = neighbor.getSharedMovies(user) / 
                           neighbor.getTotalMovies()
                sum += neighborRank * edgeWeight
            
            newRanks[user.id] = (1-damping)/numUsers + 
                               damping * sum
        
        // Check convergence
        if maxDifference(ranks, newRanks) < epsilon:
            break
            
        ranks = newRanks.copy()
    
    return ranks

function calculateUserSimilarity(user1, user2):
    // Get ratings vectors
    ratings1 = user1.getRatings()
    ratings2 = user2.getRatings()
    
    // Calculate cosine similarity
    dotProduct = 0
    norm1 = 0
    norm2 = 0
    
    for each movieId in ratings1.keys():
        if movieId in ratings2:
            dotProduct += ratings1[movieId] * ratings2[movieId]
        norm1 += ratings1[movieId] * ratings1[movieId]
    
    for each rating in ratings2.values():
        norm2 += rating * rating
    
    if norm1 == 0 or norm2 == 0:
        return 0
        
    return dotProduct / (sqrt(norm1) * sqrt(norm2))

function calculateMovieSimilarity(movie1, movie2):
    // Genre similarity (Jaccard)
    genreIntersection = movie1.genres.intersection(movie2.genres)
    genreUnion = movie1.genres.union(movie2.genres)
    genreSim = genreIntersection.size() / genreUnion.size()
    
    // Metadata similarity
    ratingDiff = abs(movie1.rating - movie2.rating) / 10.0
    yearDiff = abs(movie1.year - movie2.year) / 4.0
    lengthDiff = abs(movie1.length - movie2.length) / 180.0
    
    // Weighted combination
    return 0.6 * genreSim +
           0.2 * (1 - ratingDiff) +
           0.1 * (1 - yearDiff) +
           0.1 * (1 - lengthDiff)
``` 

### 4. Utils: Core Similarity Metrics

The Utils class provides similarity metrics used throughout the recommendation system.

#### Class Structure

```pseudocode
class Utils
    // Core functions
    static function jaccardSimilarity(set1, set2)
    static function cosineSimilarity(vector1, vector2)
    static function normalizeValue(value, min, max)
    static function calculateGenreOverlap(genres1, genres2)
    static function weightedAverage(values, weights)
```

#### Implementation Details

1. **Jaccard Similarity**:

```cpp
float jaccardSimilarity(const set<string>& set1, 
                       const set<string>& set2) {
    if (set1.empty() && set2.empty()) return 1.0f;
    
    // Calculate intersection size
    vector<string> intersection;
    set_intersection(set1.begin(), set1.end(),
                    set2.begin(), set2.end(),
                    back_inserter(intersection));
                    
    // Calculate union size
    vector<string> union_set;
    set_union(set1.begin(), set1.end(),
              set2.begin(), set2.end(),
              back_inserter(union_set));
              
    return static_cast<float>(intersection.size()) /
           static_cast<float>(union_set.size());
}
```

1. **Cosine Similarity**:

```cpp
float cosineSimilarity(const vector<float>& v1,
                      const vector<float>& v2) {
    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}
```

1. **Value Normalization**:

```cpp
float normalizeValue(float value, float min, float max) {
    if (max == min) return 0.5f;  // Handle edge case
    return (value - min) / (max - min);
}
```

#### Usage Examples

1. **Genre Similarity**:

```cpp
// Compare movie genres
auto genres1 = {"Action", "Adventure", "Sci-Fi"};
auto genres2 = {"Action", "Adventure", "Fantasy"};
float similarity = Utils::jaccardSimilarity(genres1, genres2);
// similarity = 0.667 (2 matching out of 3 total unique)
```

1. **Rating Pattern Similarity**:

```cpp
// Compare user rating patterns
vector<float> user1_ratings = {4.5, 3.0, 5.0, 2.5};
vector<float> user2_ratings = {4.0, 3.5, 4.8, 2.0};
float similarity = Utils::cosineSimilarity(
    user1_ratings, user2_ratings
);
// similarity = 0.989 (very similar rating patterns)
```

1. **Attribute Normalization**:

```cpp
// Normalize movie length
float length = 165;  // minutes
float normalized = Utils::normalizeValue(length, 60, 240);
// normalized = 0.583 (scaled to [0,1] range)
```

The utility class ensures:
- Consistent similarity calculations across the system
- Efficient reuse of common mathematical operations
- Proper handling of edge cases and normalization
- Type-safe and numerically stable implementations

### 5. Hybrid: Final Recommendation Generation

The Hybrid class represents the filal product of the recommendation system, combining content-based, collaborative, and PageRank-weighted recommendations into a final, personalized list of movie suggestions.

#### Class Structure

```pseudocode
class Hybrid
    // Core components
    content: const Content&              // Content-based filtering
    collaborative: const Collaborative&  // Collaborative filtering
    pageRank: const PageRank&           // User influence scores
    
    // Configuration
    COLLAB_BASE_WEIGHT = 0.6
    CONTENT_BASE_WEIGHT = 0.4
    MIN_SCORE_THRESHOLD = 0.1
    
    // Core functions
    function getRecommendations(userId, n = 10)
    function calculateHybridScore(contentScore, collabScore, pageRankScore)
    function normalizeScores(recommendations)
```

#### Implementation Details

1. **Score Combination**:

```cpp
float calculateHybridScore(float contentScore, 
                         float collabScore,
                         float userPageRank) {
    // Calculate adjusted weights
    float collabWeight = COLLAB_BASE_WEIGHT * (1 + userPageRank);
    float contentWeight = CONTENT_BASE_WEIGHT;
    
    // Normalize weights
    float totalWeight = collabWeight + contentWeight;
    collabWeight /= totalWeight;
    contentWeight /= totalWeight;
    
    // Combine scores
    return collabWeight * collabScore + 
           contentWeight * contentScore;
}
```

1. **Recommendation Generation**:

```cpp
vector<pair<int, float>> getRecommendations(userId, n) {
    // Get user's PageRank score
    float userRank = pageRank.getScore(userId);
    
    // Get individual recommendations
    auto contentRecs = content.getRecommendations(userId, n);
    auto collabRecs = collaborative.getRecommendations(userId, n);
    
    // Combine recommendations
    unordered_map<int, float> finalScores;
    
    for (const auto& [movieId, score] : contentRecs) {
        if (score < MIN_SCORE_THRESHOLD) continue;
        finalScores[movieId] = calculateHybridScore(
            score,
            0.0f,  // No collab score yet
            userRank
        );
    }
    
    for (const auto& [movieId, score] : collabRecs) {
        if (score < MIN_SCORE_THRESHOLD) continue;
        auto it = finalScores.find(movieId);
        if (it != finalScores.end()) {
            // Update existing score
            it->second = calculateHybridScore(
                contentRecs[movieId],
                score,
                userRank
            );
        } else {
            // Add new score
            finalScores[movieId] = calculateHybridScore(
                0.0f,  // No content score
                score,
                userRank
            );
        }
    }
    
    // Convert to vector and sort
    vector<pair<int, float>> recommendations(
        finalScores.begin(), finalScores.end()
    );
    sort(recommendations.begin(), recommendations.end(),
         [](auto& a, auto& b) { return a.second > b.second; });
         
    // Return top N
    if (recommendations.size() > n) {
        recommendations.resize(n);
    }
    return recommendations;
}
```

#### Example Flow

Let's follow a complete recommendation generation:

1. **Initial Setup**:

```cpp
// User with mixed preferences
userId = 1
userPageRank = 0.05  // Average user

// Get component recommendations
contentRecs = {
    {101, 0.85},  // Action movie
    {102, 0.75},  // Adventure movie
    {103, 0.60}   // Sci-fi movie
}

collabRecs = {
    {101, 0.90},  // Same action movie
    {104, 0.80},  // Drama movie
    {105, 0.70}   // Comedy movie
}
```

1. **Score Combination**:

```cpp
// Calculate weights
collabWeight = 0.6 * (1 + 0.05) = 0.63
contentWeight = 0.4
totalWeight = 1.03

normalizedCollabWeight = 0.63/1.03 = 0.612
normalizedContentWeight = 0.4/1.03 = 0.388

// Calculate hybrid scores
movie101 = 0.612 * 0.90 + 0.388 * 0.85 = 0.881
movie102 = 0.388 * 0.75 = 0.291
movie103 = 0.388 * 0.60 = 0.233
movie104 = 0.612 * 0.80 = 0.490
movie105 = 0.612 * 0.70 = 0.428
```

1. **Final Ranking**:

```cpp
recommendations = [
    {101, 0.881},  // Strong agreement between systems
    {104, 0.490},  // Strong collaborative signal
    {105, 0.428},  // Moderate collaborative signal
    {102, 0.291},  // Moderate content signal
    {103, 0.233}   // Weak content signal
]
```

Hybrid:

- Balanced consideration of all recommendation sources
- Proper weighting based on user engagement level
- Smooth handling of missing scores from either system
- Efficient score combination and ranking
- Adaptability to user's position in the network

## Run Time Analysis

### PageRank Component Analysis

#### Notation and Variables

- `N`: Number of users in the system
- `E`: Number of edges in the user similarity graph
- `M`: Average number of movies rated per user
- `k`: Number of iterations until convergence
- `d`: Damping factor (0.85)
- `ε`: Convergence threshold (constant, 0.0001)

#### Data Structure Creation and Initialization

1. **User Graph Construction**:

   ```cpp
   // For each user's movie ratings
   for each user in N:
       for each movie in user.ratings:  // O(M)
           for each other_user rating same movie:  // O(U_m)
               add_edge(user, other_user)
   ```

   - Time Complexity: O(N × M × U_m), where U_m is users per movie
   - Space Complexity: O(E) for adjacency lists
   - Optimization: Edge weights cached after first calculation

2. **Edge Weight Computation**:

   ```cpp
   weight(u1, u2) = shared_movies / total_movies_u2
   ```

   - Time per edge: O(M) using set intersection
   - Total preprocessing: O(E × M)
   - Space: O(E) in weight cache

#### Core Algorithm Steps

1. **Initialization**:

   ```cpp
   ranks = new Array[N].fill(1/N)  // O(N)
   newRanks = new Array[N]         // O(N)
   ```

   - Time: O(N)
   - Space: O(N) for rank vectors

2. **Iteration Process**:

   ```cpp
   for iteration in 1..k:           // O(k)
       for each user in N:          // O(N)
           sum = 0
           for each neighbor in adj[user]:  // O(deg(v))
               sum += ranks[neighbor] * weights[user,neighbor]
           newRanks[user] = (1-d)/N + d*sum
   ```

   - Time per iteration: O(E) as Σdeg(v) = 2|E|
   - Total iteration time: O(k × E)
   - Space: O(N) for rank vectors

3. **Convergence Check**:

   ```cpp
   diff = max(|newRanks[i] - ranks[i]|) for i in 1..N
   if diff < ε: break
   ```

   - Time per check: O(N)
   - Total convergence checks: O(k × N)

#### Overall Complexity

1. **Time Complexity**:
   ```
   Total = Preprocessing + Iterations
        = O(E × M) + O(k × E)
        = O(E × (M + k))
   ```
   Where:
   - `k` typically ranges from 15-20 iterations
   - `M` is usually small (10-100 ratings per user)
   - `E` is often sparse, approximately O(N × avg_degree)

2. **Space Complexity**:
   ```
   Total = Graph + Weights + Ranks
        = O(E) + O(E) + O(N)
        = O(E)
   ```

3. **Practical Considerations**:
   - Graph is typically sparse (E << N²)
   - Convergence usually occurs in <20 iterations
   - Edge weights are cached after first computation
   - Memory usage is dominated by edge storage

#### Optimizations Implemented

1. **Edge Weight Caching**:
   - First calculation: O(M) time
   - Subsequent lookups: O(1)
   - Trade-off: O(E) extra space for cache

2. **Sparse Graph Representation**:
   - Adjacency lists vs. matrix
   - Space reduced from O(N²) to O(E)
   - Iteration time from O(N²) to O(E)

3. **Early Convergence**:

   ```cpp
   if max_diff < ε:
       break  // Typically saves 5-10 iterations
   ```

4. **Parallel Processing**:
   - Edge weight computation parallelized
   - Each iteration's user updates independent
   - Theoretical speedup: O(num_threads)

#### Example Analysis

For a typical dataset with:

- N = 10,000 users
- M = 50 movies per user
- E = 100,000 edges (sparse graph)
- k = 15 iterations

```
Preprocessing:
- Edge creation: O(N × M × U_m) ≈ 10⁶ operations
- Weight computation: O(E × M) ≈ 5 × 10⁶ operations

Core Algorithm:
- Per iteration: O(E) = 100,000 operations
- Total iterations: O(k × E) = 1.5 × 10⁶ operations

Memory Usage:
- Graph: O(E) ≈ 800 KB (8 bytes per edge)
- Weights: O(E) ≈ 400 KB (4 bytes per weight)
- Ranks: O(N) ≈ 80 KB (8 bytes per user)
```

This analysis shows:
1. Preprocessing dominates the runtime
2. Memory usage scales linearly with edges
3. Iteration count has minimal impact

### Content-Based Component Analysis

#### Notation and Variables
- `N`: Number of users
- `M`: Number of movies in the system
- `G`: Average number of genres per movie
- `R`: Average number of ratings per user
- `C`: Cache size limit (constant, 10000)

#### Data Structure Creation and Initialization

1. **Similarity Cache Setup**:

   ```cpp
   similarityCache: HashMap<uint64, float>  // O(1) lookup
   cacheAccessCount: HashMap<uint64, int>   // For LRU tracking
   ```
   - Space Complexity: O(C) where C is cache size
   - Initialization Time: O(1)

2. **Genre Index Creation**:

   ```cpp
   // For each movie's genres
   for each movie in M:
       for each genre in movie.genres:  // O(G)
           genreIndex[genre].push_back(movie)
   ```
   - Time Complexity: O(M × G)
   - Space Complexity: O(M × G)

#### Core Operations

1. **Movie Similarity Calculation**:

   ```cpp
   float calculateSimilarity(movie1, movie2):
       // Genre similarity (Jaccard)
       intersection = genres1 ∩ genres2     // O(G)
       union = genres1 ∪ genres2           // O(G)
       genreSim = |intersection| / |union|
       
       // Metadata similarity
       ratingDiff = |rating1 - rating2|    // O(1)
       yearDiff = |year1 - year2|          // O(1)
       lengthDiff = |length1 - length2|    // O(1)
       
       return weighted_combination         // O(1)
   ```
   - Time per calculation: O(G)
   - Cache hit lookup: O(1)
   - Space per entry: O(1)

2. **Similarity Precomputation**:

   ```cpp
   void preComputeSimilarities():
       for each movie1 in M:
           for each movie2 in M:
               if not cached(movie1, movie2):
                   sim = calculateSimilarity(movie1, movie2)
                   cache(movie1, movie2, sim)
   ```
   - Time Complexity: O(M² × G)
   - Parallelized: O((M² × G)/num_threads)
   - Space: O(C) limited by cache size

3. **Recommendation Generation**:

   ```cpp
   vector<pair<int,float>> getRecommendations(userId):
       // Get user's genre preferences
       preferences = analyzePreferences(userId)  // O(R × G)
       
       // Score unwatched movies
       for each movie in M:
           if not watched(userId, movie):
               score = calculateScore(movie, preferences)  // O(G)
               recommendations.push(movie, score)
       
       sort(recommendations)  // O(M log M)
       return top_n(recommendations)  // O(1)
   ```
   - Time Complexity: O(R × G + M × G + M log M)
   - Space Complexity: O(M) for recommendations

#### Overall Complexity

1. **Time Complexity**:
   ```
   Preprocessing = O(M² × G)  // One-time cost
   Per Request = O(R × G + M × G + M log M)

   Realworld Application = O(1) // processing done periodically,
   // retrieve data from cache only
   // if rec. options run out, retrieve recs. from similar users with high pagerank.
   // this is the idea behind pagerank, unfortunately I haven't implemented it yet
   ```

2. **Space Complexity**:
   ```
   Total = Cache + Indices + Working
        = O(C) + O(M × G) + O(M)
        = O(M × G)  // As C is constant
   ```

3. **Optimizations**:
   - Similarity caching reduces common calculations
   - Genre index speeds up preference matching
   - Parallel precomputation of similarities
   - Early filtering of low-potential movies

### Collaborative Component Analysis

#### Notation and Variables
- `N`: Number of users
- `M`: Number of movies
- `R`: Average ratings per user
- `K`: Number of nearest neighbors used
- `P`: User's PageRank score (0 to 1)

#### Data Structure Creation

1. **User Rating Matrix**:

   ```cpp
   // Sparse matrix representation
   unordered_map<int, unordered_map<int, float>> userRatings;
   ```
   - Space: O(N × R)
   - Construction: O(N × R)

2. **Similarity Cache**:

   ```cpp
   struct CacheEntry {
       float similarity;
       int accessCount;
   };
   unordered_map<uint64, CacheEntry> cache;
   ```
   - Space: O(C) where C is cache size
   - Lookup: O(1)

#### Core Operations

1. **User Similarity Calculation**:

   ```cpp
   float calculateSimilarity(user1, user2):
       // Create rating vectors
       map1 = hashmap(user1.ratings)     // O(R)
       
       // Calculate cosine similarity
       for each rating in user2.ratings:  // O(R)
           if movie in map1:
               dotProduct += rating × map1[movie]
               norm2 += rating²
       
       return dotProduct / (norm1 × norm2)
   ```
   - Time: O(R)
   - Space: O(R)

2. **Neighbor Selection**:

   ```cpp
   vector<User> findNeighbors(userId):
       similarities = []
       for each other in users:          // O(N)
           sim = getSimilarity(userId, other)  // O(1) cached
           if sim > threshold:
               similarities.push(other, sim)
       
       sort(similarities)                // O(N log N)
       return top_k(similarities)        // O(K)
   ```
   - Time: O(N log N)
   - Space: O(N)

3. **Score Prediction**:

   ```cpp
   float predictScore(userId, movieId):
       neighbors = findNeighbors(userId)  // O(N log N)
       
       weightedSum = 0
       totalWeight = 0
       for each neighbor in neighbors:    // O(K)
           if rated(neighbor, movieId):
               weight = similarity × pagerank
               weightedSum += weight × rating
               totalWeight += weight
       
       return weightedSum / totalWeight
   ```
   - Time: O(N log N + K)
   - Space: O(K)

#### Overall Complexity

1. **Time Complexity**:
   ```
   Preprocessing = O(N² × R)  // Similarity computation
   Per Request = O(N log N + K × M)
   ```

2. **Space Complexity**:
   ```
   Total = Ratings + Cache + Working
        = O(N × R) + O(C) + O(K)
        = O(N × R)  // Typically dominates
   ```

### Hybrid Component Analysis

#### Notation and Variables
- All previous variables plus:
- `α`: Collaborative weight (0.6 base)
- `β`: Content weight (0.4 base)

#### Core Operations

1. **Weight Calculation**:

   ```cpp
   pair<float,float> calculateWeights(pagerank):
       collabWeight = α × (1 + pagerank)
       contentWeight = β
       total = collabWeight + contentWeight
       return normalize(collabWeight, contentWeight)
   ```
   - Time: O(1)
   - Space: O(1)

2. **Score Combination**:

   ```cpp
   vector<Recommendation> combineScores(contentRecs, collabRecs):
       weights = calculateWeights(userPageRank)  // O(1)
       
       scores = hashmap()
       // Process content recommendations
       for each rec in contentRecs:      // O(M)
           scores[rec.id] = contentWeight × rec.score
       
       // Process collaborative recommendations
       for each rec in collabRecs:       // O(M)
           if rec.id in scores:
               scores[rec.id] += collabWeight × rec.score
           else:
               scores[rec.id] = collabWeight × rec.score
       
       // Sort final scores
       return sortByScore(scores)        // O(M log M)
   ```
   - Time: O(M log M)
   - Space: O(M)

#### Overall System Complexity

1. **Time Complexity**:

   ```
   Preprocessing = max(
       O(M² × G),    // Content similarities
       O(N² × R),    // User similarities
       O(E × (M + k))  // PageRank
   )
   
   Per Request = O(
       M × G +        // Content scoring
       N log N +      // Neighbor finding
       K × M +        // Collaborative scoring
       M log M        // Final sorting
   )
   ```

2. **Space Complexity**:

   ```
   Total = Content + Collaborative + PageRank
        = O(M × G) + O(N × R) + O(E)
   ```

3. **Practical Performance**:
   - Preprocessing done offline
   - Heavy use of caching
   - Parallel computation where possible
   - Early pruning of low-potential items

This complete analysis shows that:
1. Preprocessing is expensive but amortized
2. Per-request performance is dominated by sorting
3. Space usage is linear in primary dimensions
4. System scales well with proper optimizations

## Performance Visualizations

```text
Rough Percentage of Total Runtime
|
|  90% +----------------+
|      |                |
|  75% |                |
|      |                |    65%
|  60% |     Pre-       |    +----+
|      |  computation   |    |    |
|  45% |                |    |    |    40%
|      |                |    |    |    +----+
|  30% |                |    |    |    |    |    25%
|      |                |    |    |    |    |    +----+
|  15% |                |    |    |    |    |    |    |
|      |                |    |    |    |    |    |    |
|   0% +----------------+----+----+----+----+----+----+
       Similarity Calc       PageRank  Collab    Content
```

## Feedback

Reviewers:

- Jasper Eerkens
- Bryce Emery
- Stevie Littleton

Feedback:

- Use normalization between similarity calculations
- Consider using a Bipartite Graph for User/Items storage
- Add caching
- Use Page Rank rating as a reference key for new users to be temporarily matched with recommendations that are popular until their own preferences are generated

1. I had a lot of issues trying to build this program out without normalizing my output values from similarity calculations. This change made it possible to accurately connect each separate part of the program.

2. I ended up using a bipartite graph since it helped me envision the overarching data structure better. Things get really confusing when there's edges everywhere on a connected graph. Moreover, this improved my runtime significantly. Maybe I lost some functionality because I scrapped my Betweenness Centrality idea, but I could probably implement that on the bipartite graph as well. 

3. I added caching, because there was no other way. Runtimes are awful when you're combining so much different data. I was going to do this anyway for O(1) lookups, but this feedback helped get the ball rolling.

4. I want to use Page Rank in the future to attempt to drastically reduce runtime and make my recommendations more probabilistic. More so, this sparked an idea to use bloom filters as well, this would hopefully reduce space complexity and possibly time complexity. I did not implement this because it would make unit testing virtually impossible.

## Citations

[1] Schrage, Michael. "History's Great Recommenders: From Ancient Times to Tomorrow." MIT Press Reader, 2023. https://thereader.mitpress.mit.edu/historys-great-recommenders-from-ancient-times-to-tomorrow/

[2] Page, Lawrence, Sergey Brin, Rajeev Motwani, and Terry Winograd. "The PageRank citation ranking: Bringing order to the web." Stanford InfoLab, 1999. http://ilpubs.stanford.edu:8090/422/

[3] Goldberg, David, David Nichols, Brian M. Oki, and Douglas Terry. "Using collaborative filtering to weave an information tapestry." Communications of the ACM 35, no. 12 (1992): 61-70. https://doi.org/10.1145/138859.138867

[4] Pazzani, Michael J., and Daniel Billsus. "Content-based recommendation systems." In The adaptive web, pp. 325-341. Springer, Berlin, Heidelberg, 2007. https://doi.org/10.1007/978-3-540-72079-9_10

[5] Burke, Robin. "Hybrid recommender systems: Survey and experiments." User modeling and user-adapted interaction 12, no. 4 (2002): 331-370. https://doi.org/10.1023/A:1021240730564

[6] Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42, no. 8 (2009): 30-37. https://doi.org/10.1109/MC.2009.263

[7] Adomavicius, Gediminas, and Alexander Tuzhilin. "Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions." IEEE transactions on knowledge and data engineering 17, no. 6 (2005): 734-749. https://doi.org/10.1109/TKDE.2005.99

[8] Smith, Brent, and Greg Linden. "Two decades of recommender systems at Amazon.com." IEEE Internet Computing 21, no. 3 (2017): 12-18. https://doi.org/10.1109/MIC.2017.72

[9] Gomez-Uribe, Carlos A., and Neil Hunt. "The Netflix recommender system: Algorithms, business value, and innovation." ACM Transactions on Management Information Systems 6, no. 4 (2015): 1-19. https://doi.org/10.1145/2843948

[10] Ricci, Francesco, Lior Rokach, and Bracha Shapira, eds. "Recommender systems handbook." Springer, 2015. ISBN: 978-1-4899-7637-6
