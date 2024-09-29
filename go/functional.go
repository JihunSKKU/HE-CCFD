package heccfd

import (
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func (ctx *Context) Packing(c *Ciphertext, dataLengths []int) *Ciphertext {
	var err error
	
	if len(dataLengths) != len(c.data) {
		panic("The length of dataLengths should be the same as the length of c.data")
	}
	if c.interval != 1 {
		panic("The interval should be 1")
	}
	if c.constVal != 1 {
		panic("The constVal should be 1")
	}

	ctxts := make([]*rlwe.Ciphertext, len(dataLengths))
	indexs := make([]int, len(dataLengths))
	for i := range indexs {
		for j := 0; j < i; j++ {
			indexs[i] += dataLengths[j]
		}
	}

	wg := sync.WaitGroup{}
	wg.Add(len(dataLengths))
	for i := range dataLengths {
		go func(i int) {
			defer wg.Done()
			if ctxts[i], err = ctx.RotationNew(c.data[i], -1 * indexs[i]); err != nil {
				panic(err)
			}
		}(i)
	}
	wg.Wait()

	size := 0
	for _, l := range dataLengths {
		size += l
	}

	return &Ciphertext{
		data: 		[]*rlwe.Ciphertext{ctx.AddMany(ctxts)},
		size: 		size,
		interval: 	1,
		constVal: 	1,
		space: 	  	largestPowerOfTwoLessThan(size),
	}
}

// Rotation rotates the input ciphertext op0 by k positions and stores the result in opOut.
func (ctx *Context) Rotation(op0 *rlwe.Ciphertext, k int, opOut *rlwe.Ciphertext) (err error) {
	eval := ctx.evalPool.Get().(*hefloat.Evaluator)
	defer ctx.evalPool.Put(eval)

	rots := optimizeRotation(k, ctx.params.MaxSlots())
	if err := eval.Rotate(op0, rots[0], opOut); err != nil {
		return err
	}
	for _, r := range rots[1:] {
		if err := eval.Rotate(opOut, r, opOut); err != nil {
			return err
		}
	}
	return nil
}

// RotationNew creates a new ciphertext that is the result of rotating op0 by k positions.
func (ctx *Context) RotationNew(op0 *rlwe.Ciphertext, k int) (opOut *rlwe.Ciphertext, err error) {
	eval := ctx.evalPool.Get().(*hefloat.Evaluator)
	opOut = hefloat.NewCiphertext(*eval.GetParameters(), op0.Degree(), op0.Level())
	ctx.evalPool.Put(eval)

	if err = ctx.Rotation(op0, k, opOut); err != nil {
		return nil, err
	}

	return opOut, nil
}

// AddMany adds multiple ciphertexts together and returns the resulting ciphertext.
func (ctx *Context) AddMany(op0 []*rlwe.Ciphertext) (opOut *rlwe.Ciphertext) {
	if len(op0) == 0 {
		panic("input ciphertexts list is empty")
    }

	opOut = hefloat.NewCiphertext(ctx.params, 1, op0[0].Level())

	eval := ctx.evalPool.Get().(*hefloat.Evaluator)
	defer ctx.evalPool.Put(eval)

	for _, c := range op0 {
		if c != nil {
			eval.Add(opOut, c, opOut)
		}
	}

	return
}

func (ctx *Context) FillHEData(op0 *Ciphertext) (opOut *Ciphertext) {
	var err error
	
	opOut = op0.CopyNew()

	slots := ctx.params.MaxSlots()
	space := op0.space

	wg := sync.WaitGroup{}
	for rot := space; rot < slots; rot *= 8 {
		cList := make([][]*rlwe.Ciphertext, len(op0.data))
		for i := range cList {
			cList[i] = make([]*rlwe.Ciphertext, 8)
		}

		wg.Add(8*len(op0.data))
		for idx := range opOut.data {
			for i := 0; i < 8; i++ {
				go func(idx, i int) {
					defer wg.Done()
					if rot*i < slots {
						if cList[idx][i], err = ctx.RotationNew(opOut.data[idx], rot*i); err != nil {
							panic(err)
						}
					} else {
						cList[idx][i] = nil
					}
				}(idx, i)
			}
		}
		wg.Wait()

		wg.Add(len(op0.data))
		for idx := range opOut.data {
			go func(idx int) {
				defer wg.Done()
				opOut.data[idx] = ctx.AddMany(cList[idx])
			}(idx)
		}
		wg.Wait()
	}

	return
}