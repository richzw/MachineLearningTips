
// https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch

unsqueeze turns an n-dimensionsal tensor into an n+1-dimensional one, by adding an extra dimension of depth 1. However, since it is ambiguous which axis the new dimension should lie across (i.e. in which direction it should be "unsqueezed"), this needs to be specified by the dim argument.

Hence the resulting unsqueezed tensors have the same information, but the indices used to access them are different.

Here is a visual representation of what squeeze/unsqueeze do for an effectively 2d matrix, where it is going from a 2d tensor to a 3d one, and hence there are 3 choices for the new dimension's position:

![](https://i.stack.imgur.com/zSZ3a.png)
