def compute_lps_array(B):
    lps = [0] * len(B)
    length = 0
    i = 1
    while i < len(B):
        if B[i] == B[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(A, B):
    lps = compute_lps_array(B)
    i = 0
    j = 0
    indices = []
    while i < len(A):
        if B[j] == A[i]:
            i += 1
            j += 1
        if j == len(B):
            indices.append(i - j)
            j = lps[j - 1]
        elif i < len(A) and B[j] != A[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return indices

# Example usage
A = [1, 70, 9, 1, 2, 30, 6, 1, 2, 30, 50, 1, 2, 30]
B = [1, 2, 30]
print(kmp_search(A, B))  # Output: [3, 7]
