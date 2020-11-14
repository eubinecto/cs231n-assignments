

## questions

### Why do we need subtracting the mean?

But why are we doing this? SVM -> standardising the features is important.





### "bias trick"?

- [why should the bias treated separately?](https://stats.stackexchange.com/a/185938)




## numpy examples

### `np.hstack()`
horizontal concatenation. the number of rows must be the same.
```
import numpy as np
m1 = np.array([[1,2], [3, 4]])
m2 = np.array([[1,2], [3, 4]])
np.hstack([m1, m2])
...
array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
```