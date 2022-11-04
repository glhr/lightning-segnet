import numpy as np

def compute_output_matrix(label_max, pred_max, output_matrix=None, nclasses=None):
    if output_matrix is None:
        output_matrix = np.zeros([nclasses, 3])

    for i in range(output_matrix.shape[0]):
        temp = pred_max == i
        temp_l = label_max == i
        tp = np.logical_and(temp, temp_l)
        temp[temp_l] = True
        fp = np.logical_xor(temp, temp_l)
        temp = pred_max == i
        temp[fp] = False
        fn = np.logical_xor(temp, temp_l)
        #print(output_matrix, tp.sum())
        output_matrix[i, 0] += tp.sum()
        output_matrix[i, 1] += fp.sum()
        output_matrix[i, 2] += fn.sum()

    return output_matrix

def compute_iou(output_matrix):
    return np.sum(output_matrix[1:, 0]/(np.sum(output_matrix[1:, :], 1).astype(np.float32)+1e-10))/(output_matrix.shape[0]-1)*100
