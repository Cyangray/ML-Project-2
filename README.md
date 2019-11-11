# ML-Project-2
Project 2, course FYS-STK4155, UiO.

The PDF file Project2-report.pdf in the repository is to be considered the answer to all the five exercise in the project set. Codes have been written for exercises a, b, c and d, while the answer for exercise e is the report itself, especially the "Analysis and results" and "Conclusion" parts.

Selected results:
Selected results are in the folder selected_results, where the grid searches for the 1234 seed attached to the report are found. To obtain these:

File run_a.py, when run, reproduces figures 2 and 3 in the report.

File run_b.py, when run, reproduces the first result in table 1 in the report. To obtain all the other results, change the gradient method in line 19 to 'SGD' and 'GD'. To change the step length, change the parameter input_eta in line 22. To obtain the results for cross validation, set the boolean parameter CV in line 11 to True.

File run_c.py, when run, reproduces the grid search results found in figure 4 of the report. In order to obtain figure 5, just change 'sigmoid' in lines 82 and 85 with 'tanh'. Figures 4 and 5 correspond to the files '1234NN_6grid_e100_20-20-2_sigmoid.png' and '1234NN_6grid_e100_20-20-2_tanh.png' respectively, these to be found in selected_results/1234_seed.

File run_d.py, when run, reproduces the grid search results found in figure 8 of the report. In order to obtain figure 9, just change 'sigmoid' in lines 75, 78 and 79 with 'tanh'. Figures 8 and 9 correspond to the files '1234NN_4grid_e100_20-20-20_sigmoid_whole.png' and '1234NN_4grid_e100_20-20-20_tanh_whole.png' respectively, these to be found in selected_results/1234_seed.
