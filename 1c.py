import numpy as np

def find_pattern_occurrences(array, pattern):
    # Convert the list to a numpy array
    array = np.array(array)
    pattern = np.array(pattern)
    
    # Create a sliding window view of the array
    windows = np.lib.stride_tricks.sliding_window_view(array, len(pattern))
    
    # Find where the windows match the pattern
    matches = np.all(windows == pattern, axis=1)
    
    # Get the indices of the matches
    occurrences = np.where(matches)[0]
    
    return occurrences

# Example usage:
A = [1, 70, 9, 1, 2, 30, 6, 1, 2, 30, 50]
B = [1, 2, 30]
occurrences = find_pattern_occurrences(A, B)
print("Pattern found at indices:", occurrences)
