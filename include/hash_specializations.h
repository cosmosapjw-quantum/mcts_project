#pragma once

#include <functional>
#include <utility>
#include <vector>
#include <cstdint>

namespace std {
    // Specialization of hash for vector<int>
    template<>
    struct hash<vector<int>> {
        size_t operator()(const vector<int>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // Specialization of hash for pair<int, int>
    template<>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
}

// Utility function for combining hash values
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}