### Miscellaneous Useful Short Scripts ###


#### Visualize an Image ####

```python
import torch
from matplotlib import pyplot as plt

random_img = (torch.rand((3, 224, 320)) * 256).round()  # CxHxW
random_img = random_img.type(torch.uint8)
plt.imshow(random_img.permute(1, 2, 0))
plt.show()
```
