# KPI_analysis20220111

You can do a KPI analysis by run the main.py document.
* **list_args**: the list of data path, the size of the list should be N * 2, the first 
element of every row of this list should be the path of **ground truth** data, and the second
element fo every row of this list should be the path of **anchor** data.
* **distance_range**: the list of distance filter, the size of this list is 3*2,
[x_range, y_range, z_range], 
For example: distance_range = [[0, 30], [-10, 10], [-math.inf, math.inf]]
* **dimension**: 2D or 3D KPI analysis, should be 2 or 3
* **iou_threshold**: the threshold of iou, 0<iou_threshold<1

By running the main.py, a classification report and a confusion matrix of multi-label 
KPI analysis will be generated.
 
