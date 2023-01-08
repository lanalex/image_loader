# Installation 
```
git clone https://github.com/lanalex/image_loader.git
cd ./image_loader
pip3 install -r requirments.txt 
```

# Outline 
This is a tool that wraps several python image libraries for most:
1) Pillow
2) cv2
3) Shapely 
4) scikit-vision
All of these libraries have great functionality but to perform certein specific use cases using them , require alot of technical code with many options 
and patterns repeated in many repos. The goal this library is to wrap these common use cases in an easy to use OOP manner:

# High level goals
1) Lazy evaluation - all operations are designed to be defined but only executed on actual rendering or access.
2) Explicit coordinate system - each library has a different coordinate system (some start with top left, top right, others are center, others are widht/height while others are topl, topr, bottoml, bottomr).
    all functions explicilty define the objects and their params with the clear coordinate system.
3) Easily render images in a lazy manner inside pandas dataframe. We want to be able to add an iamge column to a pandas data frame but not store all images in memory
so that only when an image is actually dispaly, it is loaded from disk .
4) Focus on image vision tasks related to ML
5) Support serialization to pkl of that same df and deserialization 
6) For that purpose we represent the ImageeLoader and Region obejcts.
7) As such all the transformations are performed using objects that have attributes about them and are easily accessible and can be passed as params to other functions/objects
8) The API uses the best library under the hood to perform the operation in the fastest and easiest way. Sometimes cv2, sometimes pillow, sometimes scikit-image 
9) The API is use case oriented. It wraps the most common use cases in image analysis / processing so that the high level functions represent those use cases.  

for example
(this example can also be found in sample.ipynb)
```python
from seedoo.vision.utils.region import Region
from seedoo.vision.utils.image_loading import ImageLoader
import pandas as pd
path = os.path.dirname(os.path.abspath(seedoo.vision.__file__))



r1 = Region(row = 0, column = 10, width = 100, height = 100)
r2 = Region(row = 100, column = 10, width = 100, height = 100)

# If we want to 
i = ImageLoader(path = os.path.join(path, 'apple.png'))
i2 = i.draw_regions([r1,r2])
df = pd.DataFrame([{'image' : i2}])

# This is a pandas monkey patch added function 
df.render()




```