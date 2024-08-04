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

func HEConvLayer(ctx *Context, op0 *Ciphertext, layer *Conv1DLayer) (opOut *Ciphertext) {
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
	}

	cRot := make([][]*rlwe.Ciphertext, numIn)
	for i := 0; i < numIn; i++ {
		cRot[i] = make([]*rlwe.Ciphertext, layer.KernelSize)
	}

	wg := sync.WaitGroup{}
	wg.Add(numIn * layer.KernelSize)
	for i := range cRot {
		for j := range cRot[i] {
			go func(i, j int) {
				defer wg.Done()
				if cRot[i][j], err = ctx.RotationNew(op0.data[i], j); err != nil {
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
							
							weight := []float64{layer.Weight[outIdx][inIdx][k]*op0.constVal}
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

	rg := make([]sync.WaitGroup, numOut)
	wg.Add(numOut)
	for no := 0; no < numOut; no++ {
		go func(no int) {
			defer wg.Done()
			opOut.data[no] = ctx.AddMany(cOuts[no])

			partialRot := make([]*rlwe.Ciphertext, 8)
			for rot := 1; rot < chIn; rot *= 8 {
				rg[no].Add(8)
				for i := 0; i < 8; i++ {
					go func(rot, i int) {
						defer rg[no].Done()
						if rot*i < chIn {
							if partialRot[i], err = ctx.RotationNew(opOut.data[no], rot*i*space); err != nil {
								panic(err)
							}
						} else {
							partialRot[i] = nil
						}
					}(rot, i)
				}
				rg[no].Wait()

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

	opOut.space = space*(chIn/numIn)

	return opOut
}

func makeConstVector(val float64, op0 *Ciphertext) (vConst []float64) {
	vConst = []float64{val}
	vConst = append(vConst, repeatZeros(op0.interval - 1)...)
	vConst = repeatSlice(vConst, op0.size)
	vConst = append(vConst, repeatZeros(op0.space - len(vConst))...)
	vConst = repeatSlice(vConst, op0.data[0].Slots() / len(vConst))

	return
}

func HEApproxSwishLayer(ctx *Context, op0 *Ciphertext) (opOut *Ciphertext) {
	coeff4 := -0.002012 * math.Pow(op0.constVal, 4)
	opOut = &Ciphertext{
		data: 		make([]*rlwe.Ciphertext, len(op0.data)),
		size: 		op0.size,
		interval: 	op0.interval,
		constVal: 	coeff4,
		space: 		op0.space,
	}

	coeff2 := -73.2107355865 * math.Pow(op0.constVal, 2)
	coeff1 := -248.508946322 * op0.constVal
	coeff0 := -59.5427435388

	vCoeff2, vCoeff1, vCoeff0 := makeConstVector(coeff2, op0), makeConstVector(coeff1, op0), makeConstVector(coeff0, op0)

	wg := sync.WaitGroup{}
	wg.Add(len(op0.data))
	for i := range op0.data {
		go func(i int) {
			defer wg.Done()
			eval := ctx.evalPool.Get().(*hefloat.Evaluator)
			defer ctx.evalPool.Put(eval)

			op1, _ := eval.MulRelinNew(op0.data[i], op0.data[i]) // x^2
			_ = eval.Rescale(op1, op1)							
			op2, _ := eval.AddNew(op1, vCoeff2)	// x^2 + coeff2
			_ = eval.MulRelin(op1, op2, op1)	// x^2 * (x^2 + coeff2)
			_ = eval.Rescale(op1, op1)
			_ = eval.MulRelin(op0.data[i], vCoeff1, op2) // x * coeff1
			_ = eval.Rescale(op2, op2)
			_ = eval.Add(op1, op2, op1) // x^2 * (x^2 + coeff2) + x * coeff1
			_ = eval.Add(op1, vCoeff0, op1) // x^2 * (x^2 + coeff2) + x * coeff1 + coeff0
			opOut.data[i] = op1
		}(i)
	}
	wg.Wait()

	return
}

func HEFlatten(ctx *Context, op0 *Ciphertext) (opOut *Ciphertext) {
	var err error

	wg := sync.WaitGroup{}
	
	pieceCt := make([]*rlwe.Ciphertext, len(op0.data) * (op0.data[0].Slots() / op0.space))
	wg.Add(len(op0.data) * (op0.data[0].Slots() / op0.space))
	for i, ct := range op0.data {
		for j := 0; j < ct.Slots() / op0.space; j++ {
			go func(i, j int, ct *rlwe.Ciphertext) {
				idx := i*(ct.Slots() / op0.space) + j

				defer wg.Done()
				eval := ctx.evalPool.Get().(*hefloat.Evaluator)
				defer ctx.evalPool.Put(eval)

				mask := make([]float64, ct.Slots())
				for k := 0; k < ct.Slots(); k++ {
					if k >= j*op0.space && k < j*op0.space + op0.size {
						mask[k] = op0.constVal
					} else {
						mask[k] = 0
					}
				}

				pieceCt[idx], err = eval.MulRelinNew(ct, mask)
				if err != nil {
					panic(err)
				}
				if err = eval.Rescale(pieceCt[idx], pieceCt[idx]); err != nil {
					panic(err)
				}

				if err = ctx.Rotation(pieceCt[idx], j*op0.space - idx*op0.size, pieceCt[idx]); err != nil {
					panic(err)
				}
			}(i, j, ct)
		}
	}
	wg.Wait()

	opOut = &Ciphertext{
		data: []*rlwe.Ciphertext{ctx.AddMany(pieceCt)},
		size: op0.size * len(pieceCt),
		interval: 1,
		constVal: 1,
		space: op0.data[0].Slots(),
	}
	return
}

func HEFCLayer(ctx *Context, op0 *Ciphertext, layer *FCLayer) (opOut *Ciphertext) {
	var err error
	featIn := layer.InFeatures
	featOut := layer.OutFeatures
	
	q := int(math.Ceil(float64(featIn) / float64(featOut))) + 1
	newWidth := q * featOut

	matrix := make([][]float64, featOut)
	for o := 0; o < featOut; o++ {
		matrix[o] = repeatZeros(featOut-o)
		newWeight := make([]float64, len(layer.Weight[o]))
		for i := range(layer.Weight[o]) {
			newWeight[i] = layer.Weight[o][i] * op0.constVal
		}
		matrix[o] = append(matrix[o], newWeight...)
		matrix[o] = append(matrix[o], repeatZeros(newWidth - len(matrix[o]))...)
	}
	matrixT := transposeMatrix(matrix)

	ct := op0.data[0]
	cOuts := make([]*rlwe.Ciphertext, featOut)

	wg := sync.WaitGroup{}
	wg.Add(featOut)
	for o := 0; o < featOut; o++ {
		go func(o int) {
			defer wg.Done()

			vKer := []float64{}
			for i := 0; i < q; i++ {
				vKer = append(vKer, matrixT[o+featOut*i]...)
			}
			vKer = append(vKer, repeatZeros(ct.Slots()-len(vKer))...)
			vKer = rollRight(vKer, o-featOut)

			eval := ctx.evalPool.Get().(*hefloat.Evaluator)
			cOuts[o], _ = eval.MulRelinNew(ct, vKer)
			_ = eval.Rescale(cOuts[o], cOuts[o])
			ctx.evalPool.Put(eval)

			if err = ctx.Rotation(cOuts[o], o-featOut, cOuts[o]); err != nil {
				panic(err)
			}
		}(o)
	}
	wg.Wait()

	sum := ctx.AddMany(cOuts)

	// rg := sync.WaitGroup{}



	pieceCt := make([]*rlwe.Ciphertext, q)
	wg.Add(q)
	for i := 0; i < q; i++ {
		go func(i int) {
			defer wg.Done()

			if pieceCt[i], err = ctx.RotationNew(sum, i*featOut); err != nil {
				panic(err)
			}
		}(i)
	}
	
	wg.Wait()

	sum = ctx.AddMany(pieceCt)

	eval := ctx.evalPool.Get().(*hefloat.Evaluator)
	if err = eval.Add(sum, layer.Bias, sum); err != nil {
		panic(err)
	}
	ctx.evalPool.Put(eval)

	return &Ciphertext{
		data: 		[]*rlwe.Ciphertext{sum},
		size: 		featOut,
		interval: 	1,
		constVal: 	1,
		space:	 	ct.Slots(),
	}
}