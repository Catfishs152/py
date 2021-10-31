import random

def quicksort(arr, firstIndex, lastIndex):
    if firstIndex < lastIndex:
        divIndex = random_partition(arr, firstIndex, lastIndex)
        #print(divIndex)
        quicksort(arr, firstIndex, divIndex)
        quicksort(arr, divIndex + 1, lastIndex)
    else:
        return

def partition(arr, firstIndex, lastIndex):
    i = firstIndex - 1
    for j in range(firstIndex, lastIndex):
        if arr[j] <= arr[lastIndex]:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[lastIndex] = arr[lastIndex], arr[i + 1]
    return i

def random_partition(arr, firstIndex, lastIndex):
    i = random.randint(firstIndex, lastIndex)
    arr[i], arr[lastIndex] = arr[lastIndex], arr[i]
    return partition(arr, firstIndex, lastIndex)

arr = [1, 4, 7, 1, 5, 5, 3, 85, 34, 75, 23, 75, 2, 0]
print(arr[len(arr) - 1])
print("initial array:\n", arr)
quicksort(arr, 0, len(arr) - 1)
print("result array:\n", arr)