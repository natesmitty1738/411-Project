#ifndef PAGERANK_H
#define PAGERANK_H

#include <vector>
#include <unordered_map>
#include "BipartiteGraph.h"

class PageRank
{
private:
    const BipartiteGraph &graph;
    mutable std::unordered_map<int, double> ranks;

    // PageRank parameters
    static constexpr double DAMPING = 0.85;
    static constexpr int MAX_ITERATIONS = 50;
    static constexpr double CONVERGENCE_THRESHOLD = 0.0001;
    static constexpr double MIN_RANK = 0.0001;

    // Activity thresholds
    static constexpr double CORE_ACTIVITY_THRESHOLD = 0.5;
    static constexpr double ACTIVITY_BOOST = 3.0;

    // Helper methods
    void initializeRanks() const;
    void normalizeRanks(std::unordered_map<int, double> &ranks) const;
    double calculateActivityScore(size_t numRatings, size_t maxRatings) const;

public:
    explicit PageRank(const BipartiteGraph &bg);

    // Calculate and return PageRank scores
    void calculatePageRanks() const;

    // Get rank for a specific user
    double getPageRank(int userId) const;
};

#endif