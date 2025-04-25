from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], m: int, k: int) -> int:
        count = 0
        mod_map = defaultdict(int)
        mod_map[0] = 1
        total = 0

        for num in nums:
            if num % m == k:
                count += 1
            remainder = (count - k) % m
            total += mod_map[remainder]
            mod_map[count % m] += 1

        return total        