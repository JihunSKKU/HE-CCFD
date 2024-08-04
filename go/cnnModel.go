package heccfd

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

func ReadCSV(filePath string) ([][]float64, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, err
    }

    data := make([][]float64, len(records))
    for i, record := range records[1:] {
        data[i] = make([]float64, len(record))
        for j, value := range record {
            if value == "" {
                continue
            }
            data[i][j], err = strconv.ParseFloat(value, 64)
            if err != nil {
                return nil, err
            }
        }
    }
    return data, nil
}

type HEccfdModel struct {
	Conv1 *Conv1DLayer `json:"conv1"`
	Conv2 *Conv1DLayer `json:"conv2"`

	FC1 *FCLayer `json:"fc1"`
	FC2 *FCLayer `json:"fc2"`
}

func (model HEccfdModel)CCFDForward(input [][]float64) []float64 {
	// Step 1: Convolutional Layer 1
	op1 := Conv1DPredict(input, model.Conv1)

	// Step 2: ApproxSwish Activation Function
	op2 := ApproxSwishPredict(op1).([][]float64)

	// Step 3: Convolutional Layer 2
	op3 := Conv1DPredict(op2, model.Conv2)

	// Step 4: ApproxSwish Activation Function
	op4 := ApproxSwishPredict(op3).([][]float64)

	// Step 5: Flatten
	op5 := FlattenSlice(op4)

	// Step 6: Fully Connected Layer 1
	op6 := FCPredict(op5, model.FC1)

	// Step 7: ApproxSwish Activation Function
	op7 := ApproxSwishPredict(op6).([]float64)

	// Step 8: Fully Connected Layer 2
	op8 := FCPredict(op7, model.FC2)

	return op8
}

func (model HEccfdModel) HEccfdPredict(ctx *Context, op0 *Ciphertext) (opOut *Ciphertext, elaspedTime time.Duration) {
	// Step 0: Copy the data in op0 to op1
	baseTime := time.Now()
	op1 := ctx.FillHEData(op0)
	elaspedTime0 := time.Since(baseTime)
	fmt.Printf("Copy time:  \t   %v\n", elaspedTime0)

	// Step 1: Convolutional Layer 1
	baseTime = time.Now()
	op2 := HEConvLayer(ctx, op1, model.Conv1)
	elaspedTime1 := time.Since(baseTime)
	fmt.Printf("Conv1 time: \t   %v\n", elaspedTime1)

	// Step 2: ApproxSwish Activation Function
	baseTime = time.Now()
	op3 := HEApproxSwishLayer(ctx, op2)
	elaspedTime2 := time.Since(baseTime)
	fmt.Printf("ApproxSwish1 time: %v\n", elaspedTime2)

	// Step 3: Convolutional Layer 2
	baseTime = time.Now()
	op4 := HEConvLayer(ctx, op3, model.Conv2)
	elaspedTime3 := time.Since(baseTime)
	fmt.Printf("Conv2 time: \t   %v\n", elaspedTime3)

	// Step 4: ApproxSwish Activation Function
	baseTime = time.Now()
	op5 := HEApproxSwishLayer(ctx, op4)
	elaspedTime4 := time.Since(baseTime)
	fmt.Printf("ApproxSwish2 time: %v\n", elaspedTime4)

	// Step 5: Flatten
	baseTime = time.Now()
	op6 := HEFlatten(ctx, op5)
	elaspedTime5 := time.Since(baseTime)
	fmt.Printf("Flatten time: \t   %v\n", elaspedTime5)
	
	// Step 6: Fully Connected Layer 1
	baseTime = time.Now()
	op7 := HEFCLayer(ctx, op6, model.FC1)
	elaspedTime6 := time.Since(baseTime)
	fmt.Printf("FC1 time: \t   %v\n", elaspedTime6)

	// Step 7: ApproxSwish Activation Function
	baseTime = time.Now()
	op8 := HEApproxSwishLayer(ctx, op7)
	elaspedTime7 := time.Since(baseTime)
	fmt.Printf("ApproxSwish3 time: %v\n", elaspedTime7)

	// Step 8: Fully Connected Layer 2
	baseTime = time.Now()
	opOut = HEFCLayer(ctx, op8, model.FC2)
	elaspedTime8 := time.Since(baseTime)
	fmt.Printf("FC2 time: \t   %v\n", elaspedTime8)

	elaspedTime = elaspedTime0 + elaspedTime1 + elaspedTime2 + elaspedTime3 + elaspedTime4 + elaspedTime5 + elaspedTime6 + elaspedTime7 + elaspedTime8
	return
}

func (model HEccfdModel) LoadModelParams(filePath string) (HEccfdModel, error) {
	var modelParams HEccfdModel
	
	file, err := os.Open(filePath)
	if err != nil {
		return modelParams, err
	}
	defer file.Close()

	byteValue, err := io.ReadAll(file)
	if err != nil {
		return modelParams, err
	}

	var jsonData map[string]interface{}
	err = json.Unmarshal(byteValue, &jsonData)
	if err != nil {
		return modelParams, err
	}

	modelParams.Conv1 = parseConvLayer("conv1", jsonData, 1, 0)
	modelParams.Conv2 = parseConvLayer("conv2", jsonData, 1, 0)

	modelParams.FC1 = parseFCLayer("fc1", jsonData)
	modelParams.FC2 = parseFCLayer("fc2", jsonData)

	return modelParams, nil
}

func parseConvLayer(layerName string, jsonData map[string]interface{}, stride int, padding int) *Conv1DLayer {
	if convWeight, ok := jsonData[layerName+".weight"].([]interface{}); ok {
		conv := &Conv1DLayer{
			Stride: stride,
			Padding: padding,
		}
		conv.InChannels = len(convWeight[0].([]interface{}))
		conv.OutChannels = len(convWeight)
		conv.KernelSize = len(convWeight[0].([]interface{})[0].([]interface{}))

		for _, w1 := range convWeight {
			w1List := w1.([]interface{})
			w1Parsed := make([][]float64, len(w1List))
			for i, w2 := range w1List {
				w2List := w2.([]interface{})
				w2Parsed := make([]float64, len(w2List))
				for j, w3 := range w2List {
					w2Parsed[j] = w3.(float64)
				}
				w1Parsed[i] = w2Parsed
			}
			conv.Weight = append(conv.Weight, w1Parsed)
		}

		if bias, ok := jsonData[layerName+".bias"].([]interface{}); ok {
			conv.Bias = make([]float64, len(bias))
			for i, b := range bias {
				conv.Bias[i] = b.(float64)
			}
		}

		return conv
	}
	return nil
}

func parseFCLayer(layerName string, jsonData map[string]interface{}) *FCLayer {
    if fcWeight, ok := jsonData[layerName+".weight"].([]interface{}); ok {
        fc := &FCLayer{}
        fc.InFeatures = len(fcWeight[0].([]interface{}))
        fc.OutFeatures = len(fcWeight)

        weightData := fcWeight
        fc.Weight = make([][]float64, len(weightData))
        for i := range weightData {
            row := weightData[i].([]interface{})
            fc.Weight[i] = make([]float64, len(row))
            for j := range row {
                fc.Weight[i][j] = row[j].(float64)
            }
        }

        if bias, ok := jsonData[layerName+".bias"].([]interface{}); ok {
            fc.Bias = make([]float64, len(bias))
            for i := range bias {
                fc.Bias[i] = bias[i].(float64)
            }
        }

        return fc
    }
    return nil
}