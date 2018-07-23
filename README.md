# Arriz

*A real-time array visualization tool*

Usage:

```python
from arriz import Arriz
from time import sleep
import numpy as np

while Arriz.show('Title', np.random.random((60, 60))):
    sleep(0.1)

```

The first time `Arrize.show` is called, it creates a new
window. Subsequent calls update an existing window based
on the title text and the shape of the data. 

Alternative usage:

```python
from arriz import Arriz
import numpy as np
from time import sleep

data = (np.concatenate([np.arange(200), np.arange(200, 0, -1)])).reshape((20, 20))
window = Arriz('Waterfall', data.shape, grid_px=2)

while window.update(data):
    data = np.roll(data, 1)
    sleep(0.1)

```
