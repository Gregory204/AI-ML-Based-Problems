class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        for i in range(iterations):
            d = 2 * init
            init = init - learning_rate * d
            print(init)
        return round(init, ndigits=5)

print(Solution().get_minimizer(10, 0.01, 5))

'''
4.9
4.8020000000000005
4.70596
4.6118408
4.519603984000001
4.429211904320001
4.3406276662336
4.253815112908928
4.1687388106507495
4.085364034437735
4.08536

Final Answer: 4.08536
X is being pushed closer and closer to the origin.

y = x^2 => 2x 
x^2 derivative is 2x
'''
