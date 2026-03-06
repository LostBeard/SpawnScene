using SpawnScene.Models;
using System.Numerics;

namespace SpawnScene.Services;

/// <summary>
/// Matches features between image pairs using Hamming distance on BRIEF descriptors.
/// Implements ratio test (Lowe's) for robust matching.
/// </summary>
public class FeatureMatcher
{
    private readonly float _ratioThreshold;
    private readonly int _maxDistance;

    public FeatureMatcher(float ratioThreshold = 0.75f, int maxDistance = 64)
    {
        _ratioThreshold = ratioThreshold;
        _maxDistance = maxDistance;
    }

    /// <summary>
    /// Match features between two images using brute-force Hamming distance.
    /// Applies Lowe's ratio test for robust matching.
    /// </summary>
    public List<FeatureMatch> Match(List<ImageFeature> featuresA, List<ImageFeature> featuresB)
    {
        var matches = new List<FeatureMatch>();
        if (featuresA.Count == 0 || featuresB.Count == 0) return matches;

        for (int i = 0; i < featuresA.Count; i++)
        {
            int bestDist = int.MaxValue;
            int secondBestDist = int.MaxValue;
            int bestIdx = -1;

            for (int j = 0; j < featuresB.Count; j++)
            {
                int dist = HammingDistance(featuresA[i].Descriptor, featuresB[j].Descriptor);

                if (dist < bestDist)
                {
                    secondBestDist = bestDist;
                    bestDist = dist;
                    bestIdx = j;
                }
                else if (dist < secondBestDist)
                {
                    secondBestDist = dist;
                }
            }

            // Ratio test: best match must be significantly better than second best
            if (bestIdx >= 0 && bestDist < _maxDistance &&
                (secondBestDist == int.MaxValue || bestDist < _ratioThreshold * secondBestDist))
            {
                matches.Add(new FeatureMatch
                {
                    IndexA = i,
                    IndexB = bestIdx,
                    Distance = bestDist,
                });
            }
        }

        // Cross-check: verify matches are mutual
        return CrossCheck(matches, featuresA, featuresB);
    }

    /// <summary>
    /// Cross-check matches: a match A→B is valid only if B→A also matches.
    /// </summary>
    private List<FeatureMatch> CrossCheck(List<FeatureMatch> forwardMatches,
        List<ImageFeature> featuresA, List<ImageFeature> featuresB)
    {
        var reverseMatches = new Dictionary<int, int>(); // B index → best A index

        for (int j = 0; j < featuresB.Count; j++)
        {
            int bestDist = int.MaxValue;
            int bestIdx = -1;

            for (int i = 0; i < featuresA.Count; i++)
            {
                int dist = HammingDistance(featuresB[j].Descriptor, featuresA[i].Descriptor);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0 && bestDist < _maxDistance)
            {
                reverseMatches[j] = bestIdx;
            }
        }

        return forwardMatches
            .Where(m => reverseMatches.TryGetValue(m.IndexB, out int reverseA) && reverseA == m.IndexA)
            .ToList();
    }

    /// <summary>
    /// Compute Hamming distance between two 256-bit binary descriptors.
    /// </summary>
    private static int HammingDistance(byte[] a, byte[] b)
    {
        int dist = 0;
        for (int i = 0; i < 32; i++)
        {
            dist += BitOperations.PopCount((uint)(a[i] ^ b[i]));
        }
        return dist;
    }
}
