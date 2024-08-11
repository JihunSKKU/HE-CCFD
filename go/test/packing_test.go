package test

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func SaveCiphertextToFile(ctxt *heccfd.Ciphertext, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	if err := encoder.Encode(ctxt); err != nil {
		return err
	}

	return nil
}

func LoadCiphertextFromFile(filename string) (*heccfd.Ciphertext, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var ctxt heccfd.Ciphertext
	if err := decoder.Decode(&ctxt); err != nil {
		return nil, err
	}

	return &ctxt, nil
}

func sumInts(numbers []int) int {
	sum := 0
	for _, number := range numbers {
		sum += number
	}
	return sum
}

func TestPacking(*testing.T) {
	rand.Seed(time.Now().UnixNano())
	
	params := initParams()
	ctx := heccfd.NewContext(params)

	// Load CCFD test dataset
	testDatas, err := heccfd.ReadCSV("../../data/X_test.csv")
	if err != nil {
		panic(err)
	}
	// Random value
	randValue := randomIntInRange(0, len(testDatas))
	// Add to channel dimension
	testData := [][]float64{testDatas[randValue]}
	
	// testLabels, err := heccfd.ReadCSV("../../data/y_test.csv")
	// if err != nil {
	// 	panic(err)
	// }
	// testLabel := testLabels[randValue]

	// Data division
	features := len(testData[0])
	partialFeatures := []int{5, 7, 5, 6, 7}
	agencys := len(partialFeatures)
	if sumInts(partialFeatures) != features {
		panic("The number of features is not equal to the sum of the number of features of each agency")
	}
	agencyDatas := make([][]float64, agencys)
	agencyDatas[0] = testData[0][:partialFeatures[0]]
	for i := 0; i < agencys; i++ {
		agencyDatas[i] = testData[0][sumInts(partialFeatures[:i]):sumInts(partialFeatures[:i+1])]
	}

	// Encrypt
	ptxt := heccfd.NewPlaintext(agencyDatas)
	ctxt := ctx.Encrypt(ptxt)
	
	// Packing
	times := make([]time.Duration, 30)
	packingData := &heccfd.Ciphertext{}
	
	for i := 0; i < 30; i++ {
		baseTime := time.Now()
		packingData = ctx.Packing(ctxt, partialFeatures)
		times[i] = time.Since(baseTime)
		fmt.Println("Packing time:", times[i])
	}
	// averageTime print
	averageTime := time.Duration(0)
	for i := 0; i < 30; i++ {
		averageTime += times[i]
	}
	averageTime /= 30
	fmt.Println("Average packing time:", averageTime)

	// Compare the result
	for i := 0; i < agencys; i++ {
		fmt.Println("Before packing:", agencyDatas[i])
	}

	packingM := ctx.Decrypt(packingData)
	fmt.Println("\nAfter packing: ", packingM.GetData()[0][:features])
}

func TestSave(t *testing.T) {
	SaveCiphertextToFile(&heccfd.Ciphertext{}, "../../data/ctxt.gob")
}

func TestLoad(t *testing.T) {
	LoadCiphertextFromFile("../../data/ctxt.gob")
}