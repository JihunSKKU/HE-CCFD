package heccfd

import (
	"encoding/csv"
	"encoding/json"
	"io"
	"os"
	"strconv"
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

type HeccfdModel struct {
	Conv1 *Conv1DLayer `json:"conv1"`
	Conv2 *Conv1DLayer `json:"conv2"`

	FC1 *FCLayer `json:"fc1"`
	FC2 *FCLayer `json:"fc2"`

	BatchNorm1 *BatchNormLayer `json:"batchnorm1"`
	BatchNorm2 *BatchNormLayer `json:"batchnorm2"`
	BatchNorm3 *BatchNormLayer `json:"batchnorm3"`
}

func (model HeccfdModel) LoadModelParams(filePath string) (HeccfdModel, error) {
	var modelParams HeccfdModel
	
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

	modelParams.BatchNorm1 = parseBatchNormLayer("batchnorm1", jsonData)
	modelParams.BatchNorm2 = parseBatchNormLayer("batchnorm2", jsonData)
	modelParams.BatchNorm3 = parseBatchNormLayer("batchnorm3", jsonData)

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

func parseBatchNormLayer(layerName string, jsonData map[string]interface{}) *BatchNormLayer {
	if bnWeight, ok := jsonData[layerName+".weight"].([]interface{}); ok {
		bn := &BatchNormLayer{
			Channels: len(bnWeight),
		}

		bn.Gamma = make([]float64, len(bnWeight))
		for i, w := range bnWeight {
			bn.Gamma[i] = w.(float64)
		}

		if bnBias, ok := jsonData[layerName+".bias"].([]interface{}); ok {
			bn.Beta = make([]float64, len(bnBias))
			for i, b := range bnBias {
				bn.Beta[i] = b.(float64)
			}
		}

		return bn
	}
	return nil
}