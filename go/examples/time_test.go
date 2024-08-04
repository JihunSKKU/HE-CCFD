package examples

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func TestTime(*testing.T) {
	// Initialize parameters
	params := initParams()

	// Initialize the context
	baseTime := time.Now()
	ctx := heccfd.NewContext(params)
	fmt.Println("Context initialization time:", time.Since(baseTime))

	// // Random value
	// randValue := random.int()

	// Load CCFD test dataset
	testImages, err := heccfd.ReadCSV("../../data/X_test.csv")
	if err != nil {
		panic(err)
	}
	// Add to channel dimension
	testImage := [][]float64{testImages[0]}

	testLabels, err := heccfd.ReadCSV("../../data/y_test.csv")
	if err != nil {
		panic(err)
	}
	testLabel := testLabels[0]

	// Initialize the model
	var model heccfd.HEccfdModel
	model, err = model.LoadModelParams("../models/ApproxSwish_model.json")
	if err != nil {
		panic(err)
	}

	// Plaintext prediction
	ptxtResult := model.CCFDForward(testImage)[0]

	// Ciphertext prediction
	ptxt := heccfd.NewPlaintext(testImage)
	ctxt := ctx.Encrypt(ptxt)

	ctxt, elaspedTime := model.HEccfdPredict(ctx, ctxt)
	ctxtResult := ctx.Decrypt(ctxt).GetData()[0][0]

	// Print the elapsed time
	fmt.Println("Total time: \t  ", elaspedTime)

	// Print the result
	fmt.Println("\nPlaintext prediction: ", ptxtResult)
	fmt.Println("Ciphertext prediction:", ctxtResult)
	
	var ctxtPredict float64
	if sigmoid(ctxtResult) > 0.5 {
		ctxtPredict = 1
	} else {
		ctxtPredict = 0
	}

	fmt.Println("Real label:", testLabel[0])
	if testLabel[0] == ctxtPredict {
		fmt.Println("Prediction is Correct")
	} else {
		fmt.Println("Prediction is Incorrect")
	}
}

func BenchmarkExample(b *testing.B) {
	for i := 0; i < b.N; i++ {
		TestTime(nil)
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}