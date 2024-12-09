#ifndef CONTENT_H
#define CONTENT_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "BipartiteGraph.h"

class Content
{
private:
    const BipartiteGraph &graph;
    mutable std::unordered_map<uint64_t, float> similarityCache;
    mutable std::unordered_map<uint64_t, int> cacheAccessCount;
    mutable std::mutex cacheMutex;
    static constexpr size_t MAX_CACHE_SIZE = 10000;

    // Helper methods
    uint64_t createPairKey(int id1, int id2) const;
    void evictCache() const;
    float getCachedSimilarity(int itemId1, int itemId2) const;

public:
    explicit Content(const BipartiteGraph &bg) : graph(bg) {}

    // Calculate similarity between items
    float calculateSimilarity(int item1Id, int item2Id) const;

    // Pre-compute similarities between items
    void preComputeSimilarities(int numThreads = 4);

    // Get recommendations for a user
    std::vector<std::pair<int, float>> getRecommendations(int userId, size_t n = 10) const;

    // Get similar items (for testing)
    std::vector<std::pair<int, float>> getSimilarItems(int itemId, size_t n = 5) const;
};

#endif