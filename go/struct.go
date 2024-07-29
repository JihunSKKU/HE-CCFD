package heccfd

// type Rectangle struct {
// 	H int
// 	Length int
// }

// func NewRectangle(H, Length int) *Rectangle {
// 	return &Rectangle{H: H, Length: Length}
// }

// func (r *Rectangle) Size() int {
// 	return r.H * r.Length
// }

// func (r *Rectangle) Copy() *Rectangle {
// 	return NewRectangle(r.H, r.Length)
// }

// func (r *Rectangle) Shape() string {
// 	return fmt.Sprintf("(%d, %d)", r.H, r.Length)
// }

// type Cuboid struct {
// 	L int
// 	H int
// 	Length int
// }

// func NewCuboid(L, H, Length int) *Cuboid {
// 	return &Cuboid{L: L, H: H, Length: Length}
// }

// func (c *Cuboid) Size2D() int {
// 	return c.H * c.Length
// }

// func (c *Cuboid) Size3D() int {
// 	return c.L * c.H * c.Length
// }

// func (c *Cuboid) Copy() *Cuboid {
// 	return NewCuboid(c.L, c.H, c.Length)
// }

// func (c *Cuboid) Shape() string {
// 	return fmt.Sprintf("(%d, %d, %d)", c.L, c.H, c.Length)
// }

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
