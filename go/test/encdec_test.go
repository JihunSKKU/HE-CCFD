package heccfd_test

import (
	"fmt"
	"testing"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func encdec() {
	params := initParams()
	ctx := heccfd.NewContext(params)
	nums := []float64{-2.5580910291249533e-05, 
		-0.526458584157997, -1414.7474006616037, 
		-19192.611667394496, -28613.55891643511}
	scale := 1.0

	tmp := [][]float64{nums}
	ptxt := heccfd.NewPlaintext(tmp)
	ctxt := ctx.Encrypt(ptxt)

	for i := range nums {
		nums[i] *= scale
	}

	fmt.Println("Original numbers: ", nums)

	eval := ctx.GetEval()
	eval.MulRelin(ctxt.GetData()[0], scale, ctxt.GetData()[0])
	ctx.PutEval(eval)
	ptxt2 := ctx.Decrypt(ctxt)

	fmt.Println("Decrypted numbers: ", ptxt2.GetData()[0][:len(nums)])
}

func TestEncdec(t *testing.T) {
	encdec()
}

func BenchmarkEncdec(b *testing.B) {
	for i := 0; i < b.N; i++ {
		encdec()
	}
}