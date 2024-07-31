package heccfd

import (
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type Context struct {
	params 		hefloat.Parameters
	ecd 		*hefloat.Encoder
	kgen 		*rlwe.KeyGenerator
	sk 			*rlwe.SecretKey
	pk 			*rlwe.PublicKey
	rlk 		*rlwe.RelinearizationKey
	enc 		*rlwe.Encryptor
	dec 		*rlwe.Decryptor
	evalPool 	*sync.Pool
}

func NewContext(params hefloat.Parameters) (ctx *Context) {
	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()

	ctx = &Context{
		params: params,
		ecd: hefloat.NewEncoder(params),
		kgen: kgen,
		sk: sk,
		pk: pk,
		enc: rlwe.NewEncryptor(params, pk),
		dec: rlwe.NewDecryptor(params, sk),
	}

	if params.PCount() != 0 {
		ctx.rlk = kgen.GenRelinearizationKeyNew(sk)

		slots := params.MaxSlots()
		rots := genRots(slots)
		var galEls []uint64
		for i := 0; i < len(rots); i++ {
			galEls = append(galEls, params.GaloisElement(rots[i]))
		}
		galEls = append(galEls, params.GaloisElementForComplexConjugation())
		
		// eval 생성 부분이 15초 정도 걸림
		eval := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(
				ctx.rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

		ctx.evalPool = &sync.Pool{
			New: func() interface{} {
				if eval != nil {
					return eval.ShallowCopy()
				}
				return nil
			},
		}

		ctx.fillPool()
	}

	return
}

func (ctx *Context) fillPool() {
	numEval := 16

	wg := sync.WaitGroup{}
	list := make([]*hefloat.Evaluator, numEval)
	
	wg.Add(numEval)
	for i := 0; i < numEval; i++ {
		go func(i int) {
			defer wg.Done()
			list[i] = ctx.evalPool.Get().(*hefloat.Evaluator)
		}(i)
	}
	wg.Wait()

	wg.Add(numEval)
	for i := 0; i < numEval; i++ {
		go func(i int) {
			defer wg.Done()
			ctx.evalPool.Put(list[i])
		}(i)
	}
	wg.Wait()

	return
}

func (ctx *Context) Encrypt(ptxt *Plaintext) (ctxt *Ciphertext) {
	var err error

	// Currently, it panics when the number of features in data is larger than slots
	slots := ctx.params.MaxSlots()
	if slots < ptxt.space {
		panic("heccfd: Plaintext space is too large for the current context")
	}

	maxLevel := ctx.params.MaxLevel()
	numImgs := len(ptxt.data)

	ctxt = &Ciphertext{
		data: 		make([]*rlwe.Ciphertext, numImgs),
		size: 		ptxt.size,
		interval: 	ptxt.interval,
		constVal: 	ptxt.constVal,
		space: 		ptxt.space,
	}

	for i := 0; i < numImgs; i++ {
		encoded := hefloat.NewPlaintext(ctx.params, maxLevel)
		if err = ctx.ecd.Encode(ptxt.data[i], encoded); err != nil {
			panic(err)
		}
		if ctxt.data[i], err = ctx.enc.EncryptNew(encoded); err != nil {
			panic(err)
		}
	}
		
	return
}

func (ctx *Context) Decrypt(ctxt *Ciphertext) (ptxt *Plaintext) {
	var err error

	numCtxt := len(ctxt.data)

	ptxt = &Plaintext{
		data: 		make([][]float64, numCtxt),
		size: 		ctxt.size,
		interval: 	ctxt.interval,
		constVal: 	ctxt.constVal,
		space: 		ctxt.space,
	}

	ptxtC := make([][]complex128, numCtxt)
	for i := 0; i < numCtxt; i++ {
		decrypted := ctx.dec.DecryptNew(ctxt.data[i])
		ptxtC[i] = make([]complex128, ctxt.data[i].Slots())
		if err = ctx.ecd.Decode(decrypted, ptxtC[i]); err != nil {
			panic(err)
		}
	}

	for i := 0; i < numCtxt; i++ {
		ptxt.data[i] = make([]float64, ctxt.data[i].Slots())
		for p := 0; p < ctxt.data[i].Slots(); p++ {
			ptxt.data[i][p] = real(ptxtC[i][p])
		}
	}

	return
}