package test

import (
	"math/rand"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func initLogQ(depth, logScale int) (logQ []int) {
	logQ = make([]int, depth+1)
	logQ[0] = logScale + 10
	for i := 1; i <= depth; i++ {
		logQ[i] = logScale
	}
	return
}

func initParams() (params hefloat.Parameters) {
	const (
		logSlots 	= 13
		depth 		= 11
		logScale 	= 35		
	)

	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN: 				logSlots+1,
			LogQ: 				initLogQ(depth, logScale),
			LogP: 				[]int{61, 61},
			LogDefaultScale: 	logScale,
			RingType: 			ring.Standard,
		})
	if err != nil {
		panic(err)
	}
	return
}

func randomIntInRange(min, max int) int {
	return min + rand.Intn(max-min)
}