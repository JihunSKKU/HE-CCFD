package heccfd_test

import (
	"math"
	"reflect"

	"github.com/JihunSKKU/HE-CCFD/heccfd"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func initLogQ(depth, logScale int) (logQ []int) {
	logQ = make([]int, depth+1)
	logQ[0] = logScale + 10
	for i := 1; i <= depth; i++ {
		logQ[i] = logScale
	}
	return
}

func initParams() (params hefloat.Parameters) {
	const (
		logSlots 	= 13
		depth 		= 11
		logScale 	= 35		
	)

	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN: 				logSlots+1,
			LogQ: 				initLogQ(depth, logScale),
			LogP: 				[]int{60},
			LogDefaultScale: 	logScale,
			RingType: 			ring.Standard,
		})
	if err != nil {
		panic(err)
	}
	return
}

// conv1DPredict performs a 1D convolution operation on the input data using the specified convolutional layer parameters.
func conv1DPredict(data [][]float64, conv1dLayer *heccfd.Conv1DLayer) (output [][]float64) {
	inChannels := conv1dLayer.InChannels
	outChannels := conv1dLayer.OutChannels
	kernelSize := conv1dLayer.KernelSize
	stride := conv1dLayer.Stride
	padding := conv1dLayer.Padding
	weight := conv1dLayer.Weight
	bias := conv1dLayer.Bias

	// Calculate the output dimensions
	outputLength := (len(data[0]) - kernelSize + 2*padding) / stride + 1

	// Initialize the output
	output = make([][]float64, outChannels)
	for i := range output {
		output[i] = make([]float64, outputLength)
	}
	
	// Perform the convolution for each batch
	for oc := 0; oc < outChannels; oc++ {
		for ol := 0; ol < outputLength; ol++ {
			sum := bias[oc]
			for ic := 0; ic < inChannels; ic++ {
				for kl := 0; kl < kernelSize; kl++ {
					inX := ol*stride - padding + kl
					if inX >= 0 && inX < len(data[ic]) {
						sum += data[ic][inX] * weight[oc][ic][kl]
					}
				}
			}
			output[oc][ol] = sum
		}
	}

	return
}

func calculateApproxReLU(x float64) (float64) {
	const (
		scale = 30.0
	)
	coeff := []float64{0.0243987, 0.49096448, 1.08571579, 0.01212056, -0.69068458}
	
	x /= scale
	y := 0.0
	for i := 0; i < len(coeff); i++ {
		y += coeff[i] * math.Pow(x, float64(i))
	}
	y *= scale
	return y
}

// ApproxReLUPredict performs an approximate ReLU operation on the input data.
func ApproxReLUPredict(input interface{}) (output interface{}) {
	switch input := input.(type) {
	case [][]float64:
		outputData := make([][]float64, len(input))
		for i := range input {
			outputData[i] = make([]float64, len(input[i]))
			for j := range input[i] {
				outputData[i][j] = calculateApproxReLU(input[i][j])
			}
		}
		output = outputData
	case []float64:
		outputData := make([]float64, len(input))
		for i := range input {
			outputData[i] = calculateApproxReLU(input[i])
		}
		output = outputData
	default:
		panic("Invalid input type")
	}
	return
}

// FCPredict performs a fully connected layer operation on the input data using the specified FC layer parameters.
func FCPredict(data []float64, fcLayer *heccfd.FCLayer) (output []float64) {
	inFeatures := fcLayer.InFeatures
	outFeatures := fcLayer.OutFeatures
	weight := fcLayer.Weight
	bias := fcLayer.Bias

	// Initialize the output
	output = make([]float64, outFeatures)

	// Perform the fully connected operation for each batch
	for o := 0; o < outFeatures; o++ {
		sum := bias[o]
		for i := 0; i < inFeatures; i++ {
			sum += data[i] * weight[o][i]
		}
		output[o] = sum
	}

	return
}

// FlattenSlice flattens a multi-dimensional slice into a 1D slice.
func FlattenSlice(slice interface{}) (result []float64) {
	val := reflect.ValueOf(slice)
	if val.Kind() != reflect.Slice {
		return result
	}

	for i := 0; i < val.Len(); i++ {
		item := val.Index(i).Interface()
		if reflect.TypeOf(item).Kind() == reflect.Slice {
			result = append(result, FlattenSlice(item)...)
		} else {
			switch v := item.(type) {
			case int:
				result = append(result, float64(v))
			case float64:
				result = append(result, v)
			default:
				panic("Unsupported type")
			}
		}
	}

	return
}

// BatchNormPredict performs a batch normalization operation on the input data using the specified batch normalization layer parameters.
func BatchNormPredict(data interface{}, bnLayer *heccfd.BatchNormLayer) (output interface{}) {
	switch data := data.(type) {
	case [][]float64:
		outputData := make([][]float64, len(data))
		for i := range data {
			outputData[i] = make([]float64, len(data[i]))
			for j := range data[i] {
				outputData[i][j] = (data[i][j] - bnLayer.Weight[j]) / bnLayer.Bias[j]
			}
		}
		output = outputData
	case []float64:
		outputData := make([]float64, len(data))
		for i := range data {
			outputData[i] = (data[i] - bnLayer.Weight[i]) / bnLayer.Bias[i]
		}
		output = outputData
	default:
		panic("Invalid input type")
	}
	return
}