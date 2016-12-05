# Flip game
Predict the nim-value of flip game using neural network implemented via Theano.
## What is a flip game?
A one dimensional(1D) flip game is defined to be a `'01'` string of a given length `n`. A valid move of the flip game is flipping a consecutive `'11'` to `'00'`. Two players in turn make moves in a game until there is no valid move available. The player who makes the last move wins the game. A two dimensional(2D) flip game is defined similarly, in which a valid move is to flip a consecutive `'11'`, vertically or horizontally, to `'00'`.
## The nim-value of a flip game
By the above definition, the flip game is a [impartial game](https://en.wikipedia.org/wiki/Impartial_game), which will end in finite number of moves. By [Sprague-Grundy theorem](https://en.wikipedia.org/wiki/Sprague%E2%80%93Grundy_theorem), every filp game is equivalent to a nimber, the value of which will be called the nim-value of the game. By Sprague-Grundy theorem, the first player has a winning strategy if only if the nim-value of the game is non-zero. The nim-value of the game can be evaluated as the [mex](https://en.wikipedia.org/wiki/Mex_(mathematics)) of the nim-values of all possible configuration after one valid move. The nim-value of the game with no valid move is defined to be zero. For a detailed proof of Sprague-Grundy theorem, check [this link](http://udel.edu/%7Eshuying/nimgame.pdf).
## The model for flip game in one dimension

## The model for 2D flip game
We implemented a vanilla multi-layer perceptron `(25, 300, 60, 10)` for a 5 by 5 flip game, using `tanh` as the activation function and _cross entropy_ as the cost function. The model achieves a test error rate slightly less than 10%, which is significantly worse than the error rate we get in the 1D version. However, it is still way better than using logistic regression, which results a model with test error rate more than 50%. Adding a convolution layer to the MLP model did not give a meaningful improvement in terms of error rate. In conclusion, a more sophisticated model needs to be discovered for the classification of nim-values of 2D flip game.
