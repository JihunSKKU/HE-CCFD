package heccfd_test

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func convCal() {

	// Initialize the parameters
	params := initParams()

	// Initialize the context
	ctx := heccfd.NewContext(params)
	
	// Load test data
	testImages, err := heccfd.ReadCSV("../../data/X_test.csv")
	if err != nil {
		panic(err)
	}
	// Add to channel dimension
	testImage := [][]float64{testImages[0]}

	// testLabels, err := heccfd.ReadCSV("../../data/y_test.csv")
	// if err != nil {
	// 	panic(err)
	// }
	// testLabel := testLabels[0]

	// fmt.Println("shape:", len(testImages), len(testImages[0]))
	// inputLength := len(testImages[0])

	// Initialize the convolutional layer
	var model heccfd.HeccfdModel
	model, err = model.LoadModelParams("../models/best_ApproxReLU_model.json")
	if err != nil {
		panic(err)
	}

	ptxt := heccfd.NewPlaintext(testImage)
	ctxt := ctx.Encrypt(ptxt)

	ptxtResult := conv1DPredict(testImage, model.Conv1)
	
	baseTime := time.Now()
	ctxtResult := heccfd.HEConv1Layer(ctx, ctxt, model.Conv1)
	elaspedTime1 := time.Since(baseTime)

	// Decrypt the result
	result := ctx.Decrypt(ctxtResult)

	errorRate1 := float64(0)
	for i := 0; i < 32; i++ {
		for j := 0; j < 29; j++ {
			errorRate1 += math.Abs(result.GetData()[0][i*32+j] - ptxtResult[i][j])
		}
	}
	errorRate1 /= 32 * 29

	fmt.Println("Elapsed Time1:", elaspedTime1)
	fmt.Println("Error Rate:", errorRate1)
}

func TestConvCal(t *testing.T) {
	convCal()
}

func BenchmarkConvCal(b *testing.B) {
	for i := 0; i < b.N; i++ {
		convCal()
	}
}