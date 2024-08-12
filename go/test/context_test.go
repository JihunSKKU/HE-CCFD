package test

import (
	"testing"

	"github.com/JihunSKKU/HE-CCFD/heccfd"
)

func TestSaveContext(t *testing.T) {
	// TestSaveContext tests the SaveContext function.
	params := initParams()
	ctx := heccfd.NewContext(params)

	// Save context
	err := ctx.SaveContext("../context/serialized_ctx")
	if err != nil {
		t.Fatalf("Failed to save context: %v", err)
	}	
}

func TestLoadContext(t *testing.T) {
	// TestLoadContext tests the LoadContext function.
	// Load context
	ctx, err := heccfd.LoadContext("../context/serialized_ctx")
	if err != nil {
		t.Fatalf("Failed to load context: %v", err)
	}

	// Check context
	if ctx == nil {
		t.Fatal("Context is nil")
	}

	// Print key sizes
	ctx.PrintKeySizes()
}

