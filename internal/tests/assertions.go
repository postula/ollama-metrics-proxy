package tests

import (
	"testing"
)

func Expect(t *testing.T, got, expected any) {
	if got != expected {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func ExpectNil(t *testing.T, got any, msg string) {
	if got != nil {
		t.Errorf("Expected nil, got %v: %s", got, msg)
	}
}

func ExpectNotNil(t *testing.T, got any, msg string) {
	if got == nil {
		t.Errorf("Expected non-nil value: %s", msg)
	}
}

func ExpectError(t *testing.T, err any, expectedErr error) {
	if err == nil {
		t.Error("Expected error, but got nil")
		return
	}
	if err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, err)
	}
}
