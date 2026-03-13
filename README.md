# RM_revolutions_example

** Examples will only work with PyORBIT 11.1.1 or later versions **

This repository will provide and example to prepare the dataset and fit the Rossiter McLaughlin effect in both the "classical" way and through the RM revolutions
To be updated soon with documentation and references

- classical RM: standard RM
- original RM revolutions: the dataset is created following the prescriptions of Bourrier et al. ... 
- improved RM revolutions: you only need approximate time of transit and transit duratino to create the out-of-transit CCF 

LD coefficients for HARPS are mostly random. For your target, you can get the proper coefficinets using LDtk 

```python
from ldtk import BoxcarFilter
filter_harpn = BoxcarFilter('harpn',383,693)
```
