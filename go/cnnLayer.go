package heccfd

import (
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

type Conv1DLayer struct {
    InChannels  int
    OutChannels int
    KernelSize  int
    Stride      int
    Padding     int
    Weight      [][][]float64   // 3D Slice for Weight
    Bias        []float64       // Bias
}

type FCLayer struct {
    InFeatures  int
    OutFeatures int
    Weight      [][]float64
    Bias        []float64
}

type BatchNormLayer struct {
	Channels int
	Weight   []float64
	Bias     []float64
}

func HEConv1Layer(ctx *Context, op0 *Ciphertext, layer *Conv1DLayer) (opOut *Ciphertext) {
	var err error

	chIn := layer.InChannels
	chOut := layer.OutChannels
	space := op0.space
	slots := ctx.params.MaxSlots()

	opOut = &Ciphertext{
		data: 		make([]*rlwe.Ciphertext, chOut),
		size: 		(op0.size + 2*layer.Padding - layer.KernelSize) / layer.Stride + 1,
		interval: 	op0.interval * layer.Stride,
		constVal: 	1,
		space: 		space,
	}

	if chIn != 1 {
		panic("heccfd: Conv1 Input channels must be 1")
	}
	op1 := ctx.fillHEData(op0)

	cRot := make([][]*rlwe.Ciphertext, chIn)
	for i := 0; i < chIn; i++ {
		cRot[i] = make([]*rlwe.Ciphertext, layer.KernelSize)
	}

	wg := sync.WaitGroup{}
	wg.Add(chIn * layer.KernelSize)
	for i := range cRot {
		for j := range cRot[i] {
			go func(i, j int) {
				defer wg.Done()
				if cRot[i][j], err = ctx.RotationNew(op1.data[i], j); err != nil {
					panic(err)
				}
			}(i, j)
		}
	}
	wg.Wait()

	numIn := len(op1.data)
	numOut := int(math.Ceil(float64(space * chIn * chOut) / float64(slots)))

	wg.Add(numIn * numOut * layer.KernelSize)
	for o := 0; o < numOut; o++ {
		for i := 0; i < numIn; i++ {
			for k := 0; k < layer.KernelSize; k++ {
				go func(o, i, k int) {
					defer wg.Done()
					// vKer := append(
					// 	[]float64{layer.Weight[o][i][q]}
					// )



				}(o, i, k)
			}
		}
	}





	return opOut
}