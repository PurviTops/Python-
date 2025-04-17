
#1.
import numpy as np

identity_matrix = np.eye(3, dtype=float)

print(identity_matrix)

#2.
random_array = np.random.uniform(0, 1, 5)
print(random_array)

#3.
random_2d_array = np.random.randint(1, 100, size=(3, 3))
print(random_2d_array)

#4.
def custom_function(i, j):
    return i + j
# Create a 3x3 array using the custom function
custom_array = np.fromfunction(custom_function, (3, 3), dtype=int)
print(custom_array)

#5.
arr_1d = np.array([1, 2, 3, 4, 5, 6])
# Reshape it into a 2D array (2 rows, 3 columns)
arr_2d = arr_1d.reshape(2, 3)
print(arr_2d)

#6.
ones_array = np.ones((3, 3), dtype=int)  # Change dtype to float if needed
print(ones_array)

#7.
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
# Find common elements
common_items = np.intersect1d(a, b)
print(common_items)

#8.
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
# Remove elements from 'a' that are in 'b'
result = np.setdiff1d(a, b)
print(result)

#9.
a = np.arange(15)
# Limit the print output to a maximum of 6 elements
np.set_printoptions(threshold=6)
print(a)

#10.
arr = np.array([1, 2, 3, np.nan, 5, 6, 7, np.nan])
# Remove NaN values
cleaned_arr = arr[~np.isnan(arr)]
print(cleaned_arr)

#11.
array_1d = np.arange(1, 21)
print("1D Array:")
print(array_1d)
array_2d = array_1d.reshape(4, 5)
print("\n2D Array:")
print(array_2d)

#12.
array_3d = np.arange(24).reshape(2, 3, 4)
# Get shape, size, dimensions, and data type
print("Shape of array:", array_3d.shape)
print("Size of array:", array_3d.size)
print("Number of dimensions:", array_3d.ndim)
print("Data type before conversion:", array_3d.dtype)
# Convert data type to float64
array_3d_float = array_3d.astype(np.float64)
# Verify the data type change
print("Data type after conversion:", array_3d_float.dtype)

#13.
array_1d = np.arange(1, 13)
array_2d = array_1d.reshape(3, 4)
flattened_array = array_2d.ravel()
print(flattened_array)
print("\nArrays Match:", np.array_equal(array_1d, flattened_array))

#14.
a = np. array ([1, 2, 3])
b = np. array ([4, 5, 6])
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)

#15.
array_2d = np.arange(1,4).reshape(3, 1)
array_1d = np.array([4, 5, 6])

# Perform element-wise addition using broadcasting
result = array_2d + array_1d

print("2D Array (3,1):")
print(array_2d)

print("\n1D Array (3,):")
print(array_1d)

print("\nResult of Broadcasting Addition:")
print(result)

#16.
np.random.seed(42)
array_2d = np.random.randint(0, 11, size=(4, 4))  # Random integers from 0 to 10

mask = array_2d > 5

array_2d[mask] = 5

print("Original Random 2D Array:")
print(array_2d)

print("\nBoolean Mask (True where elements > 5):")
print(mask)

print("\nModified Array (All values > 5 replaced with 5):")
print(array_2d)

#17.
#np.random.seed(42)  # For reproducibility
array_4x4 = np.random.randint(0, 21, size=(4, 4))

second_row = array_4x4[1, :]

last_column = array_4x4[:, -1]

subarray_2x2 = array_4x4[:2, :2]

print("Original 4×4 Array:")
print(array_4x4)

print("\nSecond Row:")
print(second_row)

print("\nLast Column:")
print(last_column)

print("\nSubarray (First Two Rows & First Two Columns):")
print(subarray_2x2)

#18.

#19.

arr_4x4 = np.arange(1,17).reshape(4,4)
arr_4x4
eigenvalues, eigenvectors = np.linalg.eig(arr_4x4)

# Reconstruct the matrix to verify
A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print("Original Matrix A:")
print(arr_4x4)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
print("\nReconstructed Matrix (should be close to A):")
print(A_reconstructed)

#20.
arr = np.arange(1,28).reshape(3,3,3)
flattened_arr = arr.flatten()
print(flattened_array)
print("\nArrays Match:", np.array_equal(array_1d, flattened_array))

#21.
import numpy as np
import time

np.random.seed(42)  # For reproducibility
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

start_time_dot = time.time()
result_dot = np.dot(A, B)
end_time_dot = time.time()

start_time_at = time.time()
result_at = A @ B
end_time_at = time.time()

if np.allclose(result_dot, result_at):
    print("Both methods produce the same result.")
else:
    print("The results differ.")

time_dot = end_time_dot - start_time_dot
time_at = end_time_at - start_time_at

print(f"\nTime taken using np.dot(): {time_dot:.6f} seconds")
print(f"Time taken using @ operator: {time_at:.6f} seconds")

if time_at < time_dot:
    print("\nThe '@' operator is faster!")
else:
    print("\n'np.dot()' is faster!")

#22.
arr_3d = np.arange(8).reshape(2,1,4)
arr_2d = np.arange(4).reshape(4,1)

new = arr_3d + arr_2d
print(new)

#23.
arr_2d = np.random.uniform(0,1,(2,2))
print(arr_2d)
mark = arr_2d<0.5
print(mark)
arr_2d[mark]=arr_2d[mark]**2
print(arr_2d)

#24.
array_5x5 = np.arange(1, 26).reshape(5, 5)
print(array_5x5)
diagonal_elements = np.diag(array_5x5)
print(diagonal_elements)
array_5x5[2, :] = 0
print(array_5x5)
flipped_V = np.flipud(array_5x5)
print(flipped_V)
flipped_H = np.fliplr(array_5x5)
print(flipped_H)

#25.
array_4D = np.random.randint(1,100,(2,3,4,5))
print("Original 4D Array Shape:", array_4D.shape)

subarray = array_4D[0, :2, :, :]
print("\nExtracted Subarray Shape:", subarray.shape)

mean_values = np.mean(subarray, axis=2)
print("\nMean Along Axis 2:\n", mean_values)

#26.
array_10x20 = np.arange(200).reshape(10, 20)

array_20x10 = array_10x20.reshape(20, 10)
array_5x40 = array_10x20.reshape(5, 40)

print("Original Shape:", array_10x20.shape)
print("Reshaped to (20,10):", array_20x10.shape)
print("Reshaped to (5,40):", array_5x40.shape)

#27.
large_array = np.random.rand(100, 100)

reshaped_50x200 = large_array.reshape(50, 200)
reshaped_20x500 = large_array.reshape(20, 500)

print("Original Shape:", large_array.shape)
print("Reshaped to (50,200):", reshaped_50x200.shape)
print("Reshaped to (20,500):", reshaped_20x500.shape)

flattened_array = large_array.ravel()
print("Flattened Shape:", flattened_array.shape)

