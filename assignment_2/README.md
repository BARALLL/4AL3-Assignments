#### Dataset
The dataset is gathered from the [GOES data server](https://www.goes.noaa.gov/index.html) using [SunPy](https://docs.sunpy.org/en/latest/tutorial/index.html). The datasets were provided by the professor, and was used as-is for the purpose of this assignment.


For the purpose of this assignment, the Python files use the following datasets: `data-2010-15` and `data-2020-24`.
These datasets were specified for the assignment and were used to develop and evaluate the SVM model.
You can use other datasets as long as the Python file is updated accordingly.



Currently, the following files are expected:
```bash
├── data_order.npy
├── neg_class.npy
├── neg_features_historical.npy
├── neg_features_main_timechange.npy
├── neg_features_maxmin.npy
├── pos_class.npy
├── pos_features_historical.npy
├── pos_features_main_timechange.npy
└── pos_features_maxmin.npy
```
Again, the required files can be changed, but this will require further adjustments to the Python file.


#### Report

This assignment investigates the use of SVMs to study solar flares, inspired by the [2015 publication](https://arxiv.org/abs/1411.1405) by M. G. Bobra and S. Couvidat. A detailed report (`report.pdf`) summarizes the experiments and addresses the assignment questions. The project achieved a TSS score of 0.806, surpassing the original 0.761 by incorporating a larger feature set.