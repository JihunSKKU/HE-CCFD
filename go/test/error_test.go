package test

import (
	"fmt"
	"math"
	"testing"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func TestError(*testing.T) {
	// Initialize parameters
	params := initParams()

	// Initialize the context
	ctx := heccfd.NewContext(params)

	// Load CCFD test dataset
	testImages, err := heccfd.ReadCSV("../../data/X_test.csv")
	if err != nil {
		panic(err)
	}
	// Random value
	randValue := randomIntInRange(0, len(testImages))

	// Add to channel dimension
	testImage := [][]float64{testImages[randValue]}

	// Initialize the model
	var model heccfd.HEccfdModel
	model, err = model.LoadModelParams("../../models/ApproxSwish_model.json")
	if err != nil {
		panic(err)
	}

	// Ciphertext prediction
	ptxt := heccfd.NewPlaintext(testImage)
	ctxt := ctx.Encrypt(ptxt)

	// Step 0: Copy the data in op0 to op1
	op0 := ctx.FillHEData(ctxt)

	// Step 1: Convolutional Layer 1
	pt1 := heccfd.Conv1DPredict(testImage, model.Conv1)
	op1 := heccfd.HEConvLayer(ctx, op0, model.Conv1)
	result1 := ctx.Decrypt(op1).GetData()

	errorRate1 := float64(0)
	for i := 0; i < 32; i++ {
		for j := 0; j < 29; j++ {
			errorRate1 += math.Abs(result1[0][i*32+j] - pt1[i][j])
		}
	}
	errorRate1 /= 32 * 29
	fmt.Println("Conv1 Error:  \t  ", errorRate1)

	// Step 2: ApproxSwish Activation Function
	pt2 := heccfd.ApproxSwishPredict(pt1).([][]float64)
	op2 := heccfd.HEApproxSwishLayer(ctx, op1)
	result2 := ctx.Decrypt(op2).GetData()

	errorRate2 := float64(0)
	for i := 0; i < 32; i++ {
		for j := 0; j < 29; j++ {
			errorRate2 += math.Abs(result2[0][i*32+j]*op2.GetConst() - pt2[i][j])
		}
	}
	errorRate2 /= 32 * 29
	fmt.Println("ApproxSwish Error:", errorRate2)

	// Step 3: Convolutional Layer 2
	pt3 := heccfd.Conv1DPredict(pt2, model.Conv2)
	op3 := heccfd.HEConvLayer(ctx, op2, model.Conv2)
	result3 := ctx.Decrypt(op3).GetData()

	errorRate3 := float64(0)
	for i := 0; i < 64; i++ {
		for j := 0; j < 28; j++ {
			errorRate3 += math.Abs(result3[i/8][(i%8)*1024+j] - pt3[i][j])
		}
	}
	errorRate3 /= 64 * 28
	fmt.Println("Conv2 Error:   \t  ", errorRate3)

	// Step 4: ApproxSwish Activation Function
	pt4 := heccfd.ApproxSwishPredict(pt3).([][]float64)
	op4 := heccfd.HEApproxSwishLayer(ctx, op3)
	result4 := ctx.Decrypt(op4).GetData()

	errorRate4 := float64(0)
	for i := 0; i < 64; i++ {
		for j := 0; j < 28; j++ {
			errorRate4 += math.Abs(result4[i/8][(i%8)*1024+j]*op2.GetConst() - pt4[i][j])
		}
	}
	errorRate4 /= 64 * 28
	fmt.Println("ApproxSwish Error:", errorRate4)

	// Step 5: Flatten
	pt5 := heccfd.FlattenSlice(pt4)
	op5 := heccfd.HEFlatten(ctx, op4)
	result5 := ctx.Decrypt(op5).GetData()

	errorRate5 := float64(0)
	for i := 0; i < 64*28; i++ {
		errorRate5 += math.Abs(result5[0][i] - pt5[i])
	}
	errorRate5 /= 64 * 28
	fmt.Println("Flatten Error:\t  ", errorRate5)

	// Step 6: Fully Connected Layer 1
	pt6 := heccfd.FCPredict(pt5, model.FC1)
	op6 := heccfd.HEFCLayer(ctx, op5, model.FC1)
	result6 := ctx.Decrypt(op6).GetData()

	errorRate6 := float64(0)
	for i := 0; i < 64; i++ {
		errorRate6 += math.Abs(result6[0][i] - pt6[i])
	}
	errorRate6 /= 64
	fmt.Println("FC1 Error:\t  ", errorRate6)

	// Step 7: ApproxSwish Activation Function
	pt7 := heccfd.ApproxSwishPredict(pt6).([]float64)
	op7 := heccfd.HEApproxSwishLayer(ctx, op6)
	result7 := ctx.Decrypt(op7).GetData()

	errorRate7 := float64(0)
	for i := 0; i < 64; i++ {
		errorRate7 += math.Abs(result7[0][i]*op2.GetConst() - pt7[i])
	}
	errorRate7 /= 64
	fmt.Println("ApproxSwish Error:", errorRate7)

	// Step 8: Fully Connected Layer 2
	pt8 := heccfd.FCPredict(pt7, model.FC2)
	op8 := heccfd.HEFCLayer(ctx, op7, model.FC2)
	result8 := ctx.Decrypt(op8).GetData()

	errorRate8 := float64(0)
	for i := 0; i < 1; i++ {
		errorRate8 += math.Abs(result8[0][i] - pt8[i])
	}
	errorRate8 /= 1
	fmt.Println("FC2 Error:\t  ", errorRate8)
}

func BenchmarkError(b *testing.B) {
	for i := 0; i < b.N; i++ {
		TestError(nil)
	}	
}