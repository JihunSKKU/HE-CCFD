package heccfd

import (
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
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

	numIn := len(op0.data)
	numOut := int(math.Ceil(float64(space * chIn * chOut) / float64(slots)))

	opOut = &Ciphertext{
		data: 		make([]*rlwe.Ciphertext, numOut),
		size: 		(op0.size + 2*layer.Padding - layer.KernelSize) / layer.Stride + 1,
		interval: 	op0.interval * layer.Stride,
		constVal: 	1,
		space: 		space,
	}

	if chIn != 1 {
		panic("heccfd: Conv1 Input channels must be 1")
	}
	op1 := ctx.fillHEData(op0)

	cRot := make([][]*rlwe.Ciphertext, numIn)
	for i := 0; i < numIn; i++ {
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

	cOuts := make([][]*rlwe.Ciphertext, numOut)
	for i := 0; i < numOut; i++ {
		cOuts[i] = make([]*rlwe.Ciphertext, numIn * layer.KernelSize)
	}

	wg.Add(numIn * numOut * layer.KernelSize)
	for no := 0; no < numOut; no++ {
		for ni := 0; ni < numIn; ni++ {
			for k := 0; k < layer.KernelSize; k++ {
				go func(no, ni, k int) {
					defer wg.Done()
					vKer := make([]float64, 0)
					for o := 0; o < chOut / numOut; o++ {
						outIdx := no * chOut / numOut + o
						for i := 0; i < chIn / numIn; i++ {
							inIdx := ni * chIn / numIn + i
							
							weight := []float64{layer.Weight[outIdx][inIdx][k]}
							weight = append(weight, repeatZeros(opOut.interval - 1)...)
							weight = repeatSlice(weight, opOut.size)
							weight = append(weight, repeatZeros(space - len(weight))...)
							vKer = append(vKer, weight...)
						}
					}
					vKer = repeatSlice(vKer, slots / len(vKer))
					
					eval := ctx.evalPool.Get().(*hefloat.Evaluator)
					ct, err := eval.MulRelinNew(cRot[ni][k], vKer)
					if err != nil {
						panic(err)
					}
					if err = eval.Rescale(ct, ct); err != nil {
						panic(err)
					}
					ctx.evalPool.Put(eval)
					
					cOuts[no][ni*layer.KernelSize+k] = ct
				}(no, ni, k)
			}
		}
	}
	wg.Wait()

	wg.Add(numOut)
	for no := 0; no < numOut; no++ {
		go func(no int) {
			defer wg.Done()
			opOut.data[no] = ctx.AddMany(cOuts[no])

			partialRot := make([]*rlwe.Ciphertext, 8)
			rg := sync.WaitGroup{}
			for rot := 1; rot < chIn; rot *= 8 {
				rg.Add(8)
				go func(rot int) {
					defer rg.Done()
					for i := 0; i < 8; i++ {
						if rot*i < chIn {
							if partialRot[i], err = ctx.RotationNew(opOut.data[no], rot*i*space); err != nil {
								panic(err)
							}
						}
					}
				}(rot)
				rg.Wait()

				opOut.data[no] = ctx.AddMany(partialRot)
			}
			
			vBias := make([]float64, 0)
			for o := 0; o < chOut / numOut; o++ {
				outIdx := no * chOut / numOut + o

				bias := []float64{layer.Bias[outIdx]}
				bias = append(bias, repeatZeros(opOut.interval - 1)...)
				bias = repeatSlice(bias, opOut.size)
				bias = append(bias, repeatZeros(space*(chIn/numIn) - len(bias))...)
				vBias = append(vBias, bias...)				
			}
			vBias = repeatSlice(vBias, slots / len(vBias))

			eval := ctx.evalPool.Get().(*hefloat.Evaluator)
			eval.Add(opOut.data[no], vBias, opOut.data[no])
			ctx.evalPool.Put(eval)
		}(no)
	}
	wg.Wait()

	return opOut
}