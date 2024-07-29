package heccfd

import "sort"

// genRots generates a list of rotations needed for a given number of slots.
// It returns a slice of integers representing the rotations.
func genRots(slots int) (rots []int) {
	if slots == 0 {
		panic("Slot of parameter cannot be zero!")
	}

	power := 1
	for power < slots {
		for r := 1; r < 8 && r*power < slots; r++{
			rots = append(rots, r*power)
			rots = append(rots, -r*power)
		}
		power <<= 3
	}

	return
}

// filterAndSortPositive filters positive numbers from the slice and returns them in descending order.
func filterAndSortPositive(slice []int) []int {
	// Filter positive numbers
	var positives []int
	for _, value := range slice {
		if value > 0 {
			positives = append(positives, value)
		}
	}

	// Sort in descending order
	sort.Slice(positives, func(i, j int) bool {
		return positives[i] > positives[j]
	})

	return positives
}

// modRange adjusts k to be within the range -slots/2 to slots/2.
func modRange(k, slots int) int {
	k = k % slots
	halfSlots := slots / 2
	if k > halfSlots {
		k -= slots
	} else if k <= -halfSlots {
		k += slots
	}
	return k
}

// optimizeRotation returns an optimized list of rotations needed to achieve a rotation by k positions.
func optimizeRotation(k, slots int) (rotations []int) {
	if k == 0 {
		return []int{0}
	}
	k = modRange(k, slots)

	rotList := filterAndSortPositive(genRots(slots))

	for k != 0 {
		sign := 1
		if k < 0 {
			sign = -1
			k = -k
		}

		var low, high int
		high = rotList[0]
		for _, low = range rotList {
			if low < k {
				break
			}
			high = low
		}
		
		if k-low < high-k {
			rotations = append(rotations, low*sign)
			k -= low
		} else {
			rotations = append(rotations, high*sign)
			k -= high
		}
		k *= sign
	}

	return
}

// largestPowerOfTwoLessThan returns the largest power of two less than n.
func largestPowerOfTwoLessThan(n int) int {
	if n <= 1 {
		return 1
	}

	power := 1
	for power < n {
		power <<= 1
	}

	return power
}

// repeatFloat returns a slice of float64 values, all set to the given value, repeated count times.
func repeatFloat(value float64, count int) []float64 {
	if count <= 0 {
		return []float64{}
	}
	result := make([]float64, count)
	for i := range result {
		result[i] = value
	}
	return result
}

// repeatSlice repeats a slice of float64 values count times.
func repeatSlice(value []float64, count int) []float64 {
	if count <= 0 {
		return []float64{}
	}
	result := make([]float64, len(value)*count)
	for i := 0; i < count; i++ {
		copy(result[i*len(value):], value)
	}

	return result
}

// repeatZeros returns a slice of float64 values, all set to 0, repeated count times.
func repeatZeros(count int) []float64 {
	return repeatFloat(0, count)
}

// rollRight performs a right rotation on a slice of float64 values by shift positions.
func rollRight(arr []float64, shift int) []float64 {
	n := len(arr)
	shift = ((shift % n) + n) % n
	return append(arr[n-shift:], arr[:n-shift]...)
}

// transposeMatrix transposes a 2D slice of float64 values (matrix).
func transposeMatrix(matrix [][]float64) [][]float64 {
	xl := len(matrix[0])
	yl := len(matrix)
	result := make([][]float64, xl)
	for i := range result {
		result[i] = make([]float64, yl)
	}
	for i := 0; i < xl; i++ {
		for j := 0; j < yl; j++ {
			result[i][j] = matrix[j][i]
		}
	}
	return result
}